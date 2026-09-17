"""把投放计划真正跑起来,并把整场压测的观测汇总成一份报告。

报告要回答三件事,缺一件这次压测就没法用:

1. 并发到底是多少。因为到达是分布的,「配置写了 8」和「实际同时挂着 8 个」是两回事,
   所以必须从会话的起止时刻重新采样出并发曲线,给出峰值和稳态均值。
2. 有没有撑不住。看的是播放缓冲见底的比例(starvation ratio)和它的分位数尾巴;
   平均延迟好看但尾部断流严重,对实时语音来说就是不能用。
3. 失败长什么样。连接失败、握手超时、服务端 error、排队过久要分开计数。混成一个
   「错误率」会掩盖掉「容量不足」和「服务端崩了」的区别。

报告末尾给一个 pass/fail 判定,供并发扫描脚本直接读。判据集中在下面几个常量里。
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from loadsim.arrival import (
    ArrivalPlan,
    SessionLifecycle,
    build_arrival_plan,
    check_load_feasibility,
)
from loadsim.activity import FRAME_MS, SpeechPattern
from loadsim.client import SimClient, build_ssl_context
from loadsim.metrics import SessionStats, percentile

TRANSIENT_FAILURES = {
    "connect_timeout",
    "init_timeout",
    "closed_during_handshake",
    "error",
    "server_error",
}

# 排队超时不算服务端故障,单独统计。它的含义是「容量不够」而不是「服务坏了」,
# 两者要分开报:前者该加 worker,后者该查日志。混成一个失败率会把处置方向带偏。
CAPACITY_FAILURES = {"queue_timeout"}

# 判定阈值。决策延迟 p95 不得超过帧长预算(decision_budget_s,默认等于帧长):这是
# 项目里多 session 容量压测判「到顶」的判据,决策落后于帧节奏就是在积压。断流会话
# 占比 5% 沿用 burn 子命令发压机自证的口径。失败会话与排队超时都要求为零:前者是
# 服务故障,后者说明连会话都建不起来,两者任一非零都谈不上「扛得住」。
VERDICT_MAX_STARVING_SESSION_RATIO = 0.05
VERDICT_MAX_FAILED_SESSIONS = 0
VERDICT_MAX_QUEUE_TIMEOUT_RATIO = 0.0


@dataclass
class RunConfig:
    """一次压测的全部输入。写成 dataclass 是为了能直接序列化进结果文件。"""

    url: str = "ws://127.0.0.1:8006/v1/realtime?mode=audio"
    mode: str = "audio"
    arrival_mode: str = "poisson"
    target_concurrency: int = 8
    duration_s: float = 300.0
    ramp_s: Optional[float] = None
    rate: Optional[float] = None
    reuse: bool = False
    retry_mean_s: float = 3.0
    abandon_probability: float = 0.0
    seed: int = 20260915
    lifetime_mean_s: float = 60.0
    lifetime_dist: str = "lognormal"
    lifetime_min_s: float = 5.0
    lifetime_max_s: float = 600.0
    speech: Dict[str, Any] = field(default_factory=dict)
    init_payload: Dict[str, Any] = field(default_factory=dict)
    send_silence_during_pause: bool = True
    ca_file: Optional[str] = None
    insecure: bool = False
    handshake_timeout_s: float = 30.0
    concurrency_sample_s: float = 1.0
    label: str = ""
    # 上行帧长。产品前端发 1000ms 一帧,80 是 OpenAI Realtime server VAD 的分块粒度
    frame_ms: int = FRAME_MS
    # 决策延迟预算,None 时取帧长:决策慢过一帧就是落后于实时
    decision_budget_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.decision_budget_s is None:
            self.decision_budget_s = self.frame_ms / 1000.0


@dataclass
class RunReport:
    config: Dict[str, Any]
    plan: Dict[str, Any]
    summary: Dict[str, Any]
    concurrency_series: List[List[float]]
    sessions: List[Dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self) if not isinstance(self, dict) else self, indent=2, ensure_ascii=False)


def _p(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def run_load(
    cfg: RunConfig,
    *,
    frames: List[bytes],
    plan: Optional[ArrivalPlan] = None,
    on_session_done: Optional[Callable[[SessionStats], None]] = None,
) -> RunReport:
    """按计划投放会话,跑满 duration_s 后收尾。

    plan 显式传入就按它投;不传则按 cfg 里的到达参数现算一份。
    """
    if plan is None:
        plan = build_arrival_plan(
            mode=cfg.arrival_mode,
            target_concurrency=cfg.target_concurrency,
            duration_s=cfg.duration_s,
            ramp_s=cfg.ramp_s,
            rate=cfg.rate,
            reuse=cfg.reuse,
            seed=cfg.seed,
            mean_lifetime_s=cfg.lifetime_mean_s,
            lifetime_dist=cfg.lifetime_dist,
            lifetime_min_s=cfg.lifetime_min_s,
            lifetime_max_s=cfg.lifetime_max_s,
            retry_mean_s=cfg.retry_mean_s,
            abandon_probability=cfg.abandon_probability,
        )

    lifecycle = SessionLifecycle(
        mean_s=cfg.lifetime_mean_s,
        dist=cfg.lifetime_dist,
        min_s=cfg.lifetime_min_s,
        max_s=cfg.lifetime_max_s,
        seed=cfg.seed + 1,
    )
    speech_factory = lambda index: SpeechPattern(seed=cfg.seed + 1000 + index, **cfg.speech)  # noqa: E731

    t0 = time.monotonic()
    results: List[SessionStats] = []
    concurrency_series: List[List[float]] = []
    sampled_concurrency: List[float] = []
    tasks: List[asyncio.Task] = []
    running = 0
    peak_running = 0
    launched = 0
    # 会话退出时把序号推进这个队列,由投放循环消费并决定要不要补位。
    # 关键点:补位看的是「谁真的退出了」,而不是「计划里谁该退出了」。会话时长是
    # 随机抽的,计划阶段的预演和运行时的抽样不可能重合,拿预演去驱动补位会让实际
    # 并发慢慢跑偏——这正是第一版自检里并发从 8 掉到 3 的原因。
    done_queue: asyncio.Queue[int] = asyncio.Queue()

    def _on_done(task: asyncio.Task) -> None:
        nonlocal running
        running -= 1
        try:
            stats = task.result()
        except Exception as exc:  # noqa: BLE001 - 客户端本不该抛,抛了也要落到报告里
            stats = SessionStats(index=-1, scheduled_start_s=-1.0, mode=cfg.mode)
            stats.end_reason = "client_crashed"
            stats.error = f"{type(exc).__name__}: {exc}"
        results.append(stats)
        done_queue.put_nowait(1)
        if on_session_done:
            on_session_done(stats)

    ssl_context = build_ssl_context(cfg.url, cfg.insecure, cfg.ca_file)
    if ssl_context is not None:
        _p(f"TLS: {cfg.url.split('://')[0]},校验={'关闭(--insecure)' if cfg.insecure else ('启用,CA=' + (cfg.ca_file or '系统信任库'))}")

    def _launch(index: int, offset: float) -> None:
        nonlocal running, peak_running, launched
        client = SimClient(
            index=index,
            url=cfg.url,
            mode=cfg.mode,
            frames=frames,
            speech=speech_factory(index),
            lifetime_s=lifecycle.sample(),
            scheduled_start_s=offset,
            init_payload=cfg.init_payload or {"system_prompt": "You are a helpful assistant."},
            send_silence_during_pause=cfg.send_silence_during_pause,
            ssl_context=ssl_context,
            handshake_timeout_s=cfg.handshake_timeout_s,
            frame_ms=cfg.frame_ms,
        )
        task = asyncio.create_task(client.run())
        task.add_done_callback(_on_done)
        tasks.append(task)
        running += 1
        launched += 1
        peak_running = max(peak_running, running)

    _p(
        f"开始压测: {plan.mode} 到达, 目标并发 {plan.target_concurrency}, "
        f"时长 {cfg.duration_s:.0f}s, 上行 {cfg.url}"
    )
    _p(f"到达说明: {plan.notes}")

    feasibility = check_load_feasibility(
        mode=cfg.arrival_mode,
        target_concurrency=cfg.target_concurrency,
        duration_s=cfg.duration_s,
        ramp_s=cfg.ramp_s,
        mean_lifetime_s=cfg.lifetime_mean_s,
        reuse=cfg.reuse,
        n_planned_sessions=plan.n_sessions,
    )
    if feasibility:
        _p(f"警告: {feasibility}")

    sampler_task = asyncio.create_task(
        _sample_concurrency(t0, cfg.duration_s, cfg.concurrency_sample_s, lambda: running, concurrency_series, sampled_concurrency)
    )

    try:
        # 阶段一:按到达分布把爬升段投出去。补位模式下 offsets 只含爬升段。
        for index, offset in enumerate(plan.offsets):
            delay = t0 + offset - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            _launch(index, offset)
            if cfg.reuse:
                # 爬升期间退出的会话也补,免得爬坡还没到顶就开始掉
                while not done_queue.empty():
                    done_queue.get_nowait()
                    if time.monotonic() - t0 < cfg.duration_s:
                        _launch(launched, time.monotonic() - t0)

        # 阶段二:爬升结束。补位模式下等会话退出再补,把并发顶住;
        # 不补位就一直等到时长跑完。
        while True:
            remaining = t0 + cfg.duration_s - time.monotonic()
            if remaining <= 0:
                break
            if not cfg.reuse:
                await asyncio.sleep(remaining)
                break
            try:
                await asyncio.wait_for(done_queue.get(), timeout=min(remaining, 1.0))
            except asyncio.TimeoutError:
                continue
            if cfg.abandon_probability and random.random() < cfg.abandon_probability:
                continue
            delay = random.expovariate(1.0 / cfg.retry_mean_s) if cfg.retry_mean_s > 0 else 0.0
            if delay > 0:
                await asyncio.sleep(min(delay, max(0.0, t0 + cfg.duration_s - time.monotonic())))
            if time.monotonic() - t0 >= cfg.duration_s:
                break
            _launch(launched, time.monotonic() - t0)
    finally:
        sampler_task.cancel()
        await asyncio.gather(sampler_task, return_exceptions=True)

    _p(f"投放结束,还有 {running} 个会话在跑,等它们各自结束")
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    results.sort(key=lambda s: s.index)
    summary = summarize(results, cfg, plan, sampled_concurrency, peak_running)
    _p(
        f"完成: {summary['sessions_total']} 会话, 峰值并发 {summary['concurrency_peak']}, "
        f"失败 {summary['sessions_failed']}, 断流会话占比 {summary['starving_session_ratio']:.1%}, "
        f"判定 {summary['verdict']}"
    )
    return RunReport(
        config=asdict(cfg),
        plan={
            "mode": plan.mode,
            "seed": plan.seed,
            "notes": plan.notes,
            "target_concurrency": plan.target_concurrency,
            "n_sessions": plan.n_sessions,
            "sessions_launched": launched,
            "offsets_head": [round(o, 3) for o in plan.offsets[:20]],
        },
        summary=summary,
        concurrency_series=concurrency_series,
        sessions=[s.to_dict() for s in results],
    )


async def _sample_concurrency(
    t0: float,
    duration_s: float,
    interval_s: float,
    get_running: Callable[[], int],
    series: List[List[float]],
    sampled: List[float],
) -> None:
    """每秒采一次「此刻还挂着几个会话」,得到实际并发曲线。"""
    while True:
        now = time.monotonic()
        if now - t0 > duration_s:
            return
        n = get_running()
        series.append([round(now - t0, 3), float(n)])
        sampled.append(float(n))
        await asyncio.sleep(interval_s)


def summarize(
    sessions: List[SessionStats],
    cfg: RunConfig,
    plan: ArrivalPlan,
    sampled_concurrency: List[float],
    peak_running: int,
) -> Dict[str, Any]:
    """把逐会话的观测汇总成整场压测的口径。

    这里的每一项都刻意区分了「均值和分位数」:实时语音的失败几乎都发生在尾巴上,
    只报均值会把问题抹掉。
    """
    total = len(sessions)
    failed = [s for s in sessions if s.end_reason in TRANSIENT_FAILURES]
    capacity = [s for s in sessions if s.end_reason in CAPACITY_FAILURES]
    starving = [s for s in sessions if s.underrun_total_s > 0.5]

    def _vals(fn: Callable[[SessionStats], Optional[float]]) -> List[float]:
        out = []
        for s in sessions:
            v = fn(s)
            if v is not None:
                out.append(v)
        return out

    setup = _vals(lambda s: s.setup_latency_s)
    queue_wait = _vals(lambda s: s.queue_wait_s)
    init_latency = _vals(lambda s: s.init_latency_s)
    ttfa = _vals(lambda s: s.ttfa_s)
    starve_ratio = _vals(lambda s: s.starve_ratio)
    durations = _vals(lambda s: s.session_duration_s)

    steady = sampled_concurrency
    # 稳态并发:丢掉到达窗口内的爬升段,只看后半程,避免把爬升期的低并发平均进去
    tail_n = max(1, len(steady) // 2) if steady else 1
    steady_mean = float(np.mean(steady[-tail_n:])) if steady else 0.0

    # 决策延迟按帧汇总,不按会话:先算每会话分位数再取平均会把长会话与短会话等权,
    # 而容量判据问的是「所有帧里有多少比例落后」,只有把全部样本合起来才是这个口径。
    decisions = [v for s in sessions for v in s.decision_latencies]
    decode_ms = [v for s in sessions for v in s.server_decode_ms]
    budget = cfg.decision_budget_s if cfg.decision_budget_s is not None else cfg.frame_ms / 1000.0
    over_budget = sum(1 for v in decisions if v > budget)

    queue_timeout_ratio = round(len(capacity) / total, 4) if total else 0.0
    starving_ratio = round(len(starving) / total, 4) if total else 0.0
    decision_p95 = _stat(decisions, "p95")
    reasons: List[str] = []
    if len(failed) > VERDICT_MAX_FAILED_SESSIONS:
        reasons.append(f"失败会话 {len(failed)} 个,要求为 {VERDICT_MAX_FAILED_SESSIONS}")
    if queue_timeout_ratio > VERDICT_MAX_QUEUE_TIMEOUT_RATIO:
        reasons.append(f"排队超时会话占比 {queue_timeout_ratio:.1%},要求为 {VERDICT_MAX_QUEUE_TIMEOUT_RATIO:.0%}")
    if decision_p95 is None:
        reasons.append("没有收到任何带 response_id 的决策事件,决策延迟无法判定")
    elif decision_p95 > budget:
        reasons.append(f"决策延迟 p95 {decision_p95:.3f}s 超过预算 {budget:.3f}s")
    if starving_ratio > VERDICT_MAX_STARVING_SESSION_RATIO:
        reasons.append(f"断流会话占比 {starving_ratio:.1%} 超过 {VERDICT_MAX_STARVING_SESSION_RATIO:.0%}")

    return {
        "verdict": "pass" if not reasons else "fail",
        "verdict_reasons": reasons,
        "decision_budget_s": round(budget, 4),
        "decision_p50_s": _stat(decisions, "p50"),
        "decision_p95_s": decision_p95,
        "decision_p99_s": _stat(decisions, "p99"),
        "decision_max_s": round(max(decisions), 4) if decisions else None,
        "decision_over_budget_ratio": round(over_budget / len(decisions), 4) if decisions else 0.0,
        "frames_answered_total": len(decisions),
        "frames_unanswered_total": sum(s.frames_unanswered for s in sessions),
        "speak_frames_total": sum(s.speak_frames for s in sessions),
        "listen_frames_total": sum(s.listen_frames for s in sessions),
        "server_decode_ms_p50": _stat(decode_ms, "p50"),
        "server_decode_ms_p95": _stat(decode_ms, "p95"),
        "sessions_total": total,
        "sessions_planned": plan.n_sessions,
        "sessions_failed": len(failed),
        "failure_rate": round(len(failed) / total, 4) if total else 0.0,
        "failure_breakdown": _counter(s.end_reason for s in sessions),
        "sessions_queue_timed_out": len(capacity),
        "queue_timeout_ratio": queue_timeout_ratio,
        "concurrency_peak": peak_running,
        "concurrency_target": plan.target_concurrency,
        "concurrency_steady_mean": round(steady_mean, 2),
        "concurrency_overshoot": round(peak_running / plan.target_concurrency, 3) if plan.target_concurrency else 0.0,
        "starving_session_ratio": starving_ratio,
        "starve_ratio_mean": _stat(starve_ratio, "mean"),
        "starve_ratio_p50": _stat(starve_ratio, "p50"),
        "starve_ratio_p95": _stat(starve_ratio, "p95"),
        "underrun_total_s_sum": round(sum(s.underrun_total_s for s in sessions), 2),
        "setup_latency_p50_s": _stat(setup, "p50"),
        "setup_latency_p95_s": _stat(setup, "p95"),
        "queue_wait_p50_s": _stat(queue_wait, "p50"),
        "queue_wait_p95_s": _stat(queue_wait, "p95"),
        "queue_wait_max_s": round(max(queue_wait), 3) if queue_wait else None,
        "init_latency_p50_s": _stat(init_latency, "p50"),
        "init_latency_p95_s": _stat(init_latency, "p95"),
        "ttfa_p50_s": _stat(ttfa, "p50"),
        "ttfa_p95_s": _stat(ttfa, "p95"),
        "session_duration_p50_s": _stat(durations, "p50"),
        "audio_deltas_total": sum(s.audio_deltas for s in sessions),
        "text_chars_total": sum(s.text_chars for s in sessions),
        "frames_sent_total": sum(s.frames_sent for s in sessions),
        "audio_sent_s_total": round(sum(s.audio_sent_s for s in sessions), 2),
        "speech_sent_s_total": round(sum(s.speech_sent_s for s in sessions), 2),
        "speech_activity_ratio": round(
            (sum(s.speech_sent_s for s in sessions) / sum(s.audio_sent_s for s in sessions)), 4
        )
        if sum(s.audio_sent_s for s in sessions) > 0
        else 0.0,
        "send_errors_total": sum(s.send_errors for s in sessions),
    }


def _stat(values: List[float], which: str) -> Optional[float]:
    if not values:
        return None
    if which == "mean":
        return round(float(np.mean(values)), 4)
    if which == "p50":
        return round(percentile(values, 50) or 0.0, 4)
    if which == "p95":
        return round(percentile(values, 95) or 0.0, 4)
    if which == "p99":
        return round(percentile(values, 99) or 0.0, 4)
    raise ValueError(which)


def _counter(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
