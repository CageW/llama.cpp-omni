#!/usr/bin/env python3
"""多 session 实时双工压测命令行。

典型用法:

    # 8 路并发,泊松到达,跑 5 分钟
    python -m loadsim.cli run --url ws://127.0.0.1:8006/v1/realtime?mode=audio \
        --arrival poisson --concurrency 8 --duration 300 \
        --user-audio tests/cases/common/user_audio/000_user_audio0.wav \
        --out results/poisson-c8.json

    # 先本机自检:起一个假服务端,不碰 GPU,确认工具本身没问题
    python -m loadsim.cli selftest
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

from loadsim.activity import FRAME_MS, SAMPLES_PER_FRAME, load_pcm16k_mono, samples_per_frame, slice_frames
from loadsim.runner import RunConfig, run_load


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loadsim", description="多 session 实时双工压测")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="对真实服务发压")
    run.add_argument("--url", default="ws://127.0.0.1:8006/v1/realtime?mode=audio")
    run.add_argument("--mode", default="audio", choices=["audio", "video", "chat"])
    run.add_argument(
        "--arrival",
        default="poisson",
        choices=["poisson", "ramp", "burst", "trickle", "all_at_once"],
        help="会话到达模式。poisson 是排队论标准口径(指数间隔),ramp 是闭环工具的斜坡,burst 复现尖峰",
    )
    run.add_argument("--concurrency", type=int, default=8, help="目标并发(会话数);加 --reuse 时是稳态并发,不加时是一轮投放的总数")
    run.add_argument("--duration", type=float, default=300.0, help="压测时长(秒)")
    run.add_argument("--ramp", type=float, default=None, help="到达窗口长度(秒),默认时长的三分之一")
    run.add_argument("--rate", type=float, default=None, help="泊松到达率(会话/秒),默认由并发与窗口推导")
    run.add_argument("--reuse", action="store_true", help="会话退出后补位,把并发顶在目标值")
    run.add_argument("--retry-mean", type=float, default=3.0, help="补位模式下退出到重连的平均间隔(秒)")
    run.add_argument("--abandon", type=float, default=0.0, help="补位模式下用户退出后不再重连的概率(0~1)")
    run.add_argument("--seed", type=int, default=20260915, help="随机种子,固定后同一配置可复现")
    run.add_argument("--lifetime-mean", type=float, default=60.0, help="会话平均时长(秒)")
    run.add_argument(
        "--lifetime-dist",
        default="lognormal",
        choices=["lognormal", "exponential", "weibull", "fixed"],
        help="会话时长分布。真实通话是右偏的,lognormal 比 exponential 更贴近实测",
    )
    run.add_argument("--lifetime-max", type=float, default=600.0, help="会话时长上限,应不超过服务端硬上限")
    run.add_argument("--talk-min", type=float, default=0.6, help="说话段最短时长(秒)。默认值出自电话语音实测,见 activity.py")
    run.add_argument("--talk-max", type=float, default=1.8, help="说话段最长时长(秒)")
    run.add_argument("--pause-min", type=float, default=1.2, help="静音间隔最短时长(秒)")
    run.add_argument("--pause-max", type=float, default=2.6, help="静音间隔最长时长(秒)")
    run.add_argument("--barge-in", type=float, default=0.0, help="抢话概率(0~1),命中时静音截断到近零")
    run.add_argument(
        "--no-silence-frames",
        action="store_true",
        help="静音段停止上行(移动端省流量口径);默认照常上行静音帧",
    )
    run.add_argument("--user-audio", required=True, help="上行音频素材,16 kHz 单声道 16 位 WAV")
    run.add_argument(
        "--frame-ms",
        type=int,
        default=FRAME_MS,
        help="上行帧长(毫秒)。产品前端发 1000ms 一帧、模型每秒决策一次,压容量请用 1000;"
        "默认 80 是 OpenAI Realtime server VAD 的分块粒度,比真实流量多 12.5 倍决策次数",
    )
    run.add_argument(
        "--decision-budget-ms",
        type=int,
        default=None,
        help="决策延迟预算(毫秒),p95 超过它判 fail;默认等于 --frame-ms",
    )
    run.add_argument("--max-wait", type=float, default=30.0, help="排队等 worker 的上限(秒);worker 全忙时排队几十秒是正常的,压容量时要调大")
    run.add_argument("--ca-file", default=None, help="wss 用的 CA 证书;不给则用系统信任库")
    run.add_argument("--insecure", action="store_true", help="wss 不校验证书,仅用于内网临时验证")
    run.add_argument("--ref-audio", default=None, help="session.init 里的音色参考音频 WAV;不传用服务端默认音色")
    run.add_argument("--system-prompt", default="You are a helpful assistant.", help="session.init 里的系统提示词")
    run.add_argument(
        "--init-config",
        default=None,
        help="session.init 里 payload.config 的 JSON 字符串,原样透传给服务端(采样与双工参数,例如 "
        "'{\"length_penalty\":1.1,\"listen_prob_scale\":1.0}')。不传就用服务端默认",
    )
    run.add_argument("--out", default=None, help="结果 JSON 输出路径")
    run.add_argument("--label", default="", help="本次运行的标签,写进结果便于对照")

    selftest = sub.add_parser("selftest", help="用内置假服务端验证工具自身,不需要 GPU")
    selftest.add_argument("--concurrency", type=int, default=6)
    selftest.add_argument("--duration", type=float, default=20.0)
    selftest.add_argument("--user-audio", default=None, help="不传就合成一段正弦波")
    selftest.add_argument("--out", default=None)

    burn = sub.add_parser(
        "burn",
        help="发压机自证:对不做限速的本地桩压到指定并发,确认客户端自己不断流",
    )
    burn.add_argument("--concurrency", type=int, default=100, help="要验证的并发数")
    burn.add_argument("--duration", type=float, default=20.0)

    return p


def _fmt_table(report: dict) -> str:
    s = report["summary"]
    rows = [
        ("会话数(计划/实际)", f"{s['sessions_planned']} / {s['sessions_total']}"),
        ("失败会话", f"{s['sessions_failed']} ({s['failure_rate']:.1%})  {s['failure_breakdown']}"),
        (
            "排队超时会话",
            f"{s['sessions_queue_timed_out']} ({s['queue_timeout_ratio']:.1%})"
            "  —— 容量不足,不是服务故障",
        ),
        ("峰值并发 / 目标", f"{s['concurrency_peak']} / {s['concurrency_target']}  (超出 {s['concurrency_overshoot']}x)"),
        ("稳态并发均值", f"{s['concurrency_steady_mean']}"),
        ("断流会话占比", f"{s['starving_session_ratio']:.1%}"),
        ("断流时长占比 均值/中位/95分位", f"{s['starve_ratio_mean']} / {s['starve_ratio_p50']} / {s['starve_ratio_p95']}"),
        ("累计断流时长(秒)", f"{s['underrun_total_s_sum']}"),
        ("建会话延迟 中位/95分位(秒)", f"{s['setup_latency_p50_s']} / {s['setup_latency_p95_s']}"),
        ("排队等待 中位/95分位/最长(秒)", f"{s['queue_wait_p50_s']} / {s['queue_wait_p95_s']} / {s['queue_wait_max_s']}"),
        ("首帧音频延迟 中位/95分位(秒)", f"{s['ttfa_p50_s']} / {s['ttfa_p95_s']}"),
        ("上行帧总数", f"{s['frames_sent_total']}"),
        ("其中用户说话的时长(秒)", f"{s['speech_sent_s_total']} / 上行总时长 {s['audio_sent_s_total']}"),
        ("语音活动比", f"{s['speech_activity_ratio']}(电话语音实测 0.32~0.38)"),
        ("下行音频 delta 总数", f"{s['audio_deltas_total']}"),
        (
            "决策延迟 中位/95分位/99分位/最大(秒)",
            f"{_n(s['decision_p50_s'])} / {_n(s['decision_p95_s'])} / {_n(s['decision_p99_s'])} / {_n(s['decision_max_s'])}"
            f"  (预算 {s['decision_budget_s']}s)",
        ),
        ("超预算帧占比", f"{s['decision_over_budget_ratio']:.1%}"),
        ("应答帧 / 未应答帧", f"{s['frames_answered_total']} / {s['frames_unanswered_total']}"),
        ("说话帧 / 倾听帧", f"{s['speak_frames_total']} / {s['listen_frames_total']}"),
        ("服务端 decode 中位(毫秒)", f"{_n(s['server_decode_ms_p50'])}"),
        ("判定", s["verdict"] + ("" if not s["verdict_reasons"] else ":" + ";".join(s["verdict_reasons"]))),
    ]
    width = max(len(k) for k, _ in rows)
    lines = [f"  {k.ljust(width)}  {v}" for k, v in rows]
    return "\n".join(lines)


def _n(v) -> str:
    return "无" if v is None else str(v)


def _print_concurrency(report: dict, buckets: int = 12) -> None:
    series = report["concurrency_series"]
    if not series:
        return
    step = max(1, len(series) // buckets)
    sampled = series[::step][:buckets]
    peak = max((v for _, v in series), default=1) or 1
    print("\n  实际并发曲线(每秒采样,横轴为相对时刻):")
    for t, v in sampled:
        bar = "#" * int(round(v / peak * 40))
        print(f"    t={t:7.1f}s  {int(v):3d}  {bar}")


async def _cmd_run(args) -> int:
    pcm, n_frames = load_pcm16k_mono(args.user_audio)
    frame_samples = samples_per_frame(args.frame_ms)
    frames = slice_frames(pcm, frame_samples)
    if not frames:
        print(f"素材 {args.user_audio} 不足一帧(需要 {frame_samples} 采样 = {args.frame_ms}ms)", file=sys.stderr)
        return 2
    print(
        f"上行素材 {args.user_audio}: {n_frames / 16000.0:.1f}s,切成 {len(frames)} 帧 "
        f"(每帧 {frame_samples} 采样 / {args.frame_ms}ms)"
    )

    init_payload: dict = {"system_prompt": args.system_prompt}
    if args.init_config:
        init_payload["config"] = json.loads(args.init_config)
    if args.ref_audio:
        # 参考音频用来固定音色。压测里意义有两个:一是让 init 的开销与真实使用一致
        # (带参考音频的 init 要多做一次音频编码),二是不带它就等于每次压测都在用
        # 服务端默认音色,而真实流量里音色是分散的。
        ref_pcm, ref_frames = load_pcm16k_mono(args.ref_audio)
        ref_b64 = base64.b64encode(ref_pcm).decode("ascii")
        init_payload["voice"] = {"ref_audio_base64": ref_b64, "tts_ref_audio_base64": ref_b64}
        print(f"参考音频 {args.ref_audio}: {ref_frames / 16000.0:.1f}s,已随 session.init 下发")

    cfg = RunConfig(
        url=args.url,
        mode=args.mode,
        arrival_mode=args.arrival,
        target_concurrency=args.concurrency,
        duration_s=args.duration,
        ramp_s=args.ramp,
        rate=args.rate,
        reuse=args.reuse,
        retry_mean_s=args.retry_mean,
        abandon_probability=args.abandon,
        seed=args.seed,
        lifetime_mean_s=args.lifetime_mean,
        lifetime_dist=args.lifetime_dist,
        lifetime_max_s=args.lifetime_max,
        speech={
            "talk_min_s": args.talk_min,
            "talk_max_s": args.talk_max,
            "pause_min_s": args.pause_min,
            "pause_max_s": args.pause_max,
            "barge_in_probability": args.barge_in,
        },
        send_silence_during_pause=not args.no_silence_frames,
        ca_file=args.ca_file,
        insecure=args.insecure,
        handshake_timeout_s=args.max_wait,
        init_payload=init_payload,
        label=args.label,
        frame_ms=args.frame_ms,
        decision_budget_s=(args.decision_budget_ms if args.decision_budget_ms is not None else args.frame_ms) / 1000.0,
    )
    report = await run_load(cfg, frames=frames)
    _emit(report, args.out)
    return 0


def _emit(report, out: str | None) -> None:
    as_dict = {
        "config": report.config,
        "plan": report.plan,
        "summary": report.summary,
        "concurrency_series": report.concurrency_series,
        "sessions": report.sessions,
    }
    print("\n=== 压测结果 ===")
    print(_fmt_table(as_dict))
    _print_concurrency(as_dict)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(as_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n结果已写入 {path}")


def _synthetic_frames(seconds: float = 10.0) -> list[bytes]:
    """不传素材时合成一段 440 Hz 正弦,仅用于自检。"""
    import array
    import math

    total = int(16000 * seconds)
    samples = array.array("f", (0.1 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(total)))
    raw = samples.tobytes()
    frame_bytes = SAMPLES_PER_FRAME * 4
    return [raw[i : i + frame_bytes] for i in range(0, len(raw) - frame_bytes + 1, frame_bytes)]


async def _cmd_selftest(args) -> int:
    from loadsim.fake_server import serve_fake

    if args.user_audio:
        pcm, _ = load_pcm16k_mono(args.user_audio)
        frames = slice_frames(pcm, SAMPLES_PER_FRAME)
    else:
        frames = _synthetic_frames()

    server, port, stop = await serve_fake()
    print(f"假服务端已起在 127.0.0.1:{port}(按会话数线性加延迟,用于验证工具本身)")
    try:
        cfg = RunConfig(
            url=f"ws://127.0.0.1:{port}/v1/realtime?mode=audio",
            arrival_mode="ramp",
            target_concurrency=args.concurrency,
            duration_s=args.duration,
            ramp_s=args.duration * 0.25,
            reuse=True,
            lifetime_mean_s=args.duration * 0.35,
            lifetime_dist="lognormal",
            seed=12345,
            # 不覆盖说话节奏,让自检走与真实发压完全相同的默认参数
            label="selftest",
        )
        report = await run_load(cfg, frames=frames)
        _emit(report, args.out)
        peak = report.summary["concurrency_peak"]
        print(f"\n自检: 假服务端观测到峰值并发 {peak}(应接近 {args.concurrency})")
        return 0 if peak > 0 else 1
    finally:
        await stop()
        server.close()
        await server.wait_closed()


async def _cmd_burn(args) -> int:
    """发压机自证。

    对真实服务压之前先跑一次:把服务端换成一个不做任何限速的本地桩(收到就回,
    按实时节奏吐音频),同样的并发下客户端自己不该有任何断流。如果这一步就断了,
    说明瓶颈在发压机本身,后面测到的断流不能算在服务端头上。

    这一步容易被跳过,但省掉它的代价是整场压测的结论作废——而且作废得很难发现,
    因为报告里的数字看起来完全正常。
    """
    from loadsim.fake_server import serve_unthrottled

    server, port, stop = await serve_unthrottled()
    print(f"不限速桩已起在 127.0.0.1:{port},开始 {args.concurrency} 路并发自证")
    try:
        cfg = RunConfig(
            url=f"ws://127.0.0.1:{port}/v1/realtime?mode=audio",
            arrival_mode="all_at_once",
            target_concurrency=args.concurrency,
            duration_s=args.duration,
            reuse=False,
            lifetime_mean_s=args.duration,
            lifetime_dist="fixed",
            seed=3,
            speech={"pause_min_s": 0.0, "pause_max_s": 0.0},
            label="burn",
        )
        report = await run_load(cfg, frames=_synthetic_frames(5.0))
        s = report.summary
        print("\n=== 发压机自证结果 ===")
        print(f"  实际并发       {s['concurrency_peak']} / {args.concurrency}")
        print(f"  失败会话       {s['sessions_failed']}")
        print(f"  上行帧 / 下行 delta  {s['frames_sent_total']} / {s['audio_deltas_total']}")
        print(f"  建会话延迟 p95 {s['setup_latency_p95_s']}s")
        print(f"  断流会话占比   {s['starving_session_ratio']:.1%}   断流时长占比均值 {s['starve_ratio_mean']}")
        ok = s["sessions_failed"] == 0 and s["starving_session_ratio"] < 0.05
        print(f"\n判定: {'通过,这台机器可以压到 ' + str(args.concurrency) + ' 路' if ok else '未通过,瓶颈在发压机侧,请降低并发或换机器'}")
        return 0 if ok else 1
    finally:
        await stop()
        server.close()
        await server.wait_closed()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        return asyncio.run(_cmd_run(args))
    if args.cmd == "selftest":
        return asyncio.run(_cmd_selftest(args))
    if args.cmd == "burn":
        return asyncio.run(_cmd_burn(args))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
