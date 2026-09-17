"""到达模型与播放缓冲的单元测试。

这些用例钉住的是三件容易悄悄错掉、错了又看不出来的事:

- 泊松到达的间隔均值必须是 1/rate,否则压测实际压的并发和标称的对不上;
- 补位模式的实际并发必须接近目标值,第一版没做占空比补偿时只能到 82%;
- 播放缓冲的断流判定方向不能反,收到音频要加库存、墙钟走要减库存,一旦反了,
  报告里就会把正常会话报成断流、把断流报成正常。
"""

from __future__ import annotations

import math

import pytest

from loadsim.activity import FRAME_MS, SpeechActivity, SpeechPattern, samples_per_frame, slice_frames
from loadsim.arrival import (
    build_arrival_plan,
    check_load_feasibility,
    expected_sessions,
)
from loadsim.metrics import PlaybackBuffer, SessionStats, pcm_duration_s


# --------------------------------------------------------------------- 到达模型


def test_poisson_interarrival_mean_matches_rate() -> None:
    """泊松到达的平均间隔应当等于 1/rate。

    这里用大量样本而不是少量样本:指数分布方差大,样本少的时候均值波动能到几十
    个百分点,那样的断言只会随机失败,钉不住任何东西。
    """
    rate = 0.25
    plan = build_arrival_plan(
        mode="poisson", target_concurrency=8, duration_s=60000.0, ramp_s=20000.0, rate=rate, seed=1
    )
    gaps = [b - a for a, b in zip(plan.offsets, plan.offsets[1:])]
    assert len(gaps) > 2000
    assert math.isclose(sum(gaps) / len(gaps), 1.0 / rate, rel_tol=0.05)


def test_poisson_is_seed_stable() -> None:
    """同一个种子必须给出同一份计划,否则回归对比没有意义。"""
    kwargs = dict(mode="poisson", target_concurrency=6, duration_s=120.0, seed=42)
    assert build_arrival_plan(**kwargs).offsets == build_arrival_plan(**kwargs).offsets


def test_ramp_offsets_are_monotonic_and_inside_window() -> None:
    plan = build_arrival_plan(mode="ramp", target_concurrency=12, duration_s=90.0, ramp_s=30.0, seed=3)
    assert plan.offsets == sorted(plan.offsets)
    assert plan.offsets[0] >= 0.0
    assert plan.offsets[-1] <= 30.0
    assert len(plan.offsets) == 12


def test_burst_front_loads_compared_to_ramp() -> None:
    """同样会话数,突发模式的峰值必须明显高于斜坡模式,否则它就没有存在意义。"""
    burst = build_arrival_plan(mode="burst", target_concurrency=20, duration_s=120.0, ramp_s=60.0, seed=5)
    ramp = build_arrival_plan(mode="ramp", target_concurrency=20, duration_s=120.0, ramp_s=60.0, seed=5)
    burst_front = sum(1 for o in burst.offsets if o <= 3.0)
    ramp_front = sum(1 for o in ramp.offsets if o <= 3.0)
    assert burst_front >= 10
    assert ramp_front <= 2


def test_refill_oversubscribes_to_compensate_duty_cycle() -> None:
    """补位模式必须多开槽位,否则实际并发稳定低于目标值。

    占空比 = 平均时长 / (平均时长 + 补位间隔)。平均 60s、补位间隔 3s 时占空比约
    0.95,爬升人数应当略高于目标并发;时长越短、补位越慢,超配越明显。
    """
    plan = build_arrival_plan(
        mode="ramp",
        target_concurrency=10,
        duration_s=600.0,
        ramp_s=120.0,
        reuse=True,
        seed=9,
        mean_lifetime_s=30.0,
        retry_mean_s=10.0,
    )
    # 占空比 = 30 / 40 = 0.75,所以爬升阶段应当投约 14 个
    assert len(plan.offsets) >= 13


def test_refill_offsets_stay_inside_ramp_window() -> None:
    """补位模式交给运行时的只是爬升段,必须在窗口内铺完,不能拖到压测中途还在加人。"""
    plan = build_arrival_plan(
        mode="ramp",
        target_concurrency=5,
        duration_s=300.0,
        ramp_s=45.0,
        reuse=True,
        seed=11,
        mean_lifetime_s=60.0,
    )
    assert plan.offsets[-1] <= 45.0


def test_feasibility_flags_underloaded_plan() -> None:
    """不补位且到达窗口太短时,必须给出「达不到目标并发」的提示。

    这里 60 个会话铺在 20s 内,平均时长 10s,按 Little 定律稳态并发只有 30,
    低于目标 60 的 80%,应当被拦下。
    """
    msg = check_load_feasibility(
        mode="ramp",
        target_concurrency=60,
        duration_s=120.0,
        ramp_s=20.0,
        mean_lifetime_s=10.0,
        reuse=False,
        n_planned_sessions=60,
    )
    assert msg is not None
    assert "--reuse" in msg


def test_feasibility_silent_when_plan_is_adequate() -> None:
    assert (
        check_load_feasibility(
            mode="ramp",
            target_concurrency=4,
            duration_s=120.0,
            ramp_s=20.0,
            mean_lifetime_s=60.0,
            reuse=False,
            n_planned_sessions=4,
        )
        is None
    )


def test_expected_sessions_is_positive_and_grows_with_duration() -> None:
    short = expected_sessions(
        target_concurrency=8, duration_s=120.0, mean_lifetime_s=60.0, retry_mean_s=3.0, abandon_probability=0.0
    )
    long = expected_sessions(
        target_concurrency=8, duration_s=600.0, mean_lifetime_s=60.0, retry_mean_s=3.0, abandon_probability=0.0
    )
    assert short > 0 and long > short


# --------------------------------------------------------------------- 播放缓冲


def test_pcm_duration_uses_output_sample_rate() -> None:
    """下行是 24 kHz float32,一秒音频是 96000 字节。"""
    assert pcm_duration_s(96000) == pytest.approx(1.0)
    assert pcm_duration_s(0) == 0.0


def test_buffer_accumulates_then_drains_without_underrun() -> None:
    """一次性收到 2 秒音频,播到第 2 秒为止都还有货,不该断流。

    第 2 秒是边界:库存正好用尽。把「刚好播完」误判成见底是最容易犯的错,所以
    专门断言到 t=2.0 为止事件数为 0。
    """
    buf = PlaybackBuffer()
    buf.note_audio(2.0, now=0.0)
    buf.tick(1.0)
    assert buf.underrun_events == 0
    buf.tick(2.0)
    assert buf.underrun_events == 0
    buf.finish(2.0)
    assert buf.underrun_total_s == 0.0


def test_buffer_records_underrun_when_starved() -> None:
    """收到 1 秒音频却播到第 3 秒,断流从第 1 秒开始,共 2 秒。"""
    buf = PlaybackBuffer()
    buf.note_audio(1.0, now=0.0)
    buf.tick(3.0)
    assert buf.underrun_events == 1
    buf.finish(3.0)
    assert buf.underrun_total_s == pytest.approx(2.0, abs=0.01)
    assert buf.max_underrun_s == pytest.approx(2.0, abs=0.01)


def test_buffer_resumes_at_the_right_instant() -> None:
    """补货之后断流应当在欠账还清的那一刻结束。

    时间线:0s 收到 0.5s 音频,能播到 0.5s;1.5s 时已经欠了 1.0s;
    1.5s 补进 1.0s 音频,正好把欠账还平(没有余量),断流结束在 1.5s。
    这时库存为零,2.2s 时会再度断流,但那是新的一段了。
    所以到这里为止应当是 1 次事件、1.0s 断流。
    """
    buf = PlaybackBuffer()
    buf.note_audio(0.5, now=0.0)
    buf.tick(1.5)
    assert buf.underrun_events == 1
    buf.note_audio(1.0, now=1.5)
    buf.finish(1.5)
    assert buf.underrun_events == 1
    assert buf.underrun_total_s == pytest.approx(1.0, abs=0.02)


def test_buffer_counts_two_separate_underruns() -> None:
    """两段互相独立的断流要记成两次事件、时长相加,不能合成一段。

    时间线:0s 收到 0.5s,播到 0.5s 见底;到 2.0s 收到 2.0s 时已欠 1.5s,
    这些音频先还清欠账,断流在 2.0s 结束,第一段长 1.5s;
    还清后余量约 0.5s,能播到 2.5s;之后再度见底,到会话结束的 5.0s 为止,
    第二段从 2.5s 持续到 5.0s,长 2.5s。合计 4.0s,两段事件。
    """
    buf = PlaybackBuffer()
    buf.note_audio(0.5, now=0.0)
    buf.tick(2.0)
    buf.note_audio(2.0, now=2.0)
    buf.tick(3.0)
    assert buf.underrun_events == 2
    buf.tick(5.0)
    buf.finish(5.0)
    assert buf.underrun_events == 2
    assert buf.underrun_total_s == pytest.approx(3.5, abs=0.05)
    assert buf.max_underrun_s == pytest.approx(2.5, abs=0.05)


def test_steady_stream_never_underruns() -> None:
    """按实时速度持续供给音频(每 80ms 给 80ms)不该产生断流。

    这是压测报告里最重要的一条基线:如果连这个都断,说明工具的时钟算法错了,
    而不是服务端慢。
    """
    buf = PlaybackBuffer()
    now = 0.0
    for _ in range(50):
        now += 0.08
        buf.note_audio(0.08, now=now)
        buf.tick(now)
    buf.finish(now)
    assert buf.underrun_events == 0
    assert buf.underrun_total_s == 0.0


def test_slow_stream_underruns_proportionally() -> None:
    """只供给一半速度时,断流时长应当约等于总时长的一半。"""
    buf = PlaybackBuffer()
    now = 0.0
    for _ in range(20):
        now += 0.08
        buf.note_audio(0.04, now=now)
        buf.tick(now)
    buf.finish(now)
    assert buf.underrun_total_s > now * 0.35



def test_buffer_pauses_drain_when_listening() -> None:
    """收到 kind=listen 后库存暂停消耗,倾听期间不算断流。
    
    时间线:0s 收到 2s 音频,能播到 2s;1.5s 模型说完进入倾听,
    之后一直倾听到 5s,库存在 2s 被播完但此时已在倾听,不该记成断流。
    """
    buf = PlaybackBuffer()
    buf.note_audio(2.0, now=0.0)
    buf.tick(1.5)
    assert buf.underrun_events == 0
    buf.note_listen(now=1.5)
    buf.tick(5.0)
    assert buf.underrun_events == 0, "倾听时库存暂停消耗,不该断流"
    assert buf.buffered_s == pytest.approx(0.5, abs=0.01)
    buf.finish(5.0)
    assert buf.underrun_total_s == 0.0


def test_buffer_resumes_drain_after_audio() -> None:
    """收到音频后退出倾听,库存重新消耗。"""
    buf = PlaybackBuffer()
    buf.note_audio(1.0, now=0.0)
    buf.note_listen(now=0.5)
    buf.tick(3.0)
    assert buf.buffered_s == pytest.approx(0.5, abs=0.01), "倾听中库存应保持"
    buf.note_audio(0.5, now=3.0)
    buf.tick(4.5)
    buf.finish(4.5)
    assert buf.underrun_events >= 1, "退出倾听后库存耗尽应记成断流"


# --------------------------------------------------------------------- 说话节奏


def test_default_speech_pattern_matches_measured_activity_ratio() -> None:
    """默认的说话节奏要落在电话语音实测的语音活动比区间内。

    Lee & Un 1986 在 50 分钟电话语音上测得语音活动比 31.5%~37.8%,平均说话段
    1.11~1.30 秒、平均静音 2.22~2.41 秒。这个用例把默认值钉在实测附近,防止
    以后有人凭直觉改回「说话段三五秒、静音一两秒」——那种设置在同样并发下会让
    上行负载差不多翻倍,压出来的容量会明显偏悲观,而且偏得看不出来。
    """
    pattern = SpeechPattern(seed=20260915)
    activity = SpeechActivity(pattern)
    frames = 3600 * 12  # 一小时的 80ms 帧
    talking = sum(1 for _ in range(frames) if activity.advance(FRAME_MS / 1000.0))
    ratio = talking / frames
    assert 0.30 <= ratio <= 0.45, f"默认语音活动比 {ratio:.3f} 偏离实测区间 0.315~0.378 太多"


def test_default_talk_and_pause_means_are_in_measured_range() -> None:
    pattern = SpeechPattern()
    talk_mean = (pattern.talk_min_s + pattern.talk_max_s) / 2
    pause_mean = (pattern.pause_min_s + pattern.pause_max_s) / 2
    assert 1.0 <= talk_mean <= 1.5
    assert 1.5 <= pause_mean <= 3.0


# --------------------------------------------------------------------- 失败分类


def test_queue_timeout_is_not_counted_as_service_failure() -> None:
    """排队等不到 worker 是容量不足,不能算进失败率。

    这两件事的处置方向相反:容量不足要加 worker,服务故障要查日志。混成一个
    「失败率」会让读报告的人找错方向。端到端验证时实测过一次:单 worker 接 4 路
    并发,3 个会话排队 35/70/105 秒,它们全是排队,不是服务端坏了。
    """
    from loadsim.runner import CAPACITY_FAILURES, TRANSIENT_FAILURES

    assert "queue_timeout" in CAPACITY_FAILURES
    assert "queue_timeout" not in TRANSIENT_FAILURES
    # 后端侧的建会话超时才算故障
    assert "init_timeout" in TRANSIENT_FAILURES
    assert "connect_timeout" in TRANSIENT_FAILURES


def test_summary_separates_queue_timeouts_from_failures() -> None:
    """汇总里排队超时要单独一栏,且不计入 failure_rate。"""
    from loadsim.arrival import ArrivalPlan
    from loadsim.runner import RunConfig, summarize

    def _session(index: int, reason: str):
        s = SessionStats(index=index, scheduled_start_s=0.0, mode="audio")
        s.end_reason = reason
        s.connected_at = 0.0
        s.closed_at = 10.0
        return s

    sessions = [
        _session(0, "lifetime_expired"),
        _session(1, "queue_timeout"),
        _session(2, "queue_timeout"),
        _session(3, "connect_timeout"),
    ]
    plan = ArrivalPlan(offsets=[0.0] * 4, target_concurrency=4, duration_s=30.0, mode="ramp")
    cfg = RunConfig()
    summary = summarize(sessions, cfg, plan, sampled_concurrency=[4.0] * 10, peak_running=4)

    assert summary["sessions_total"] == 4
    assert summary["sessions_failed"] == 1, "只有 connect_timeout 算失败"
    assert summary["failure_rate"] == 0.25
    assert summary["sessions_queue_timed_out"] == 2
    assert summary["queue_timeout_ratio"] == 0.5


# --------------------------------------------------------------------- 帧长与决策延迟


def test_samples_per_frame_1000ms_yields_one_second_frames() -> None:
    """产品前端发 1000ms 一帧,16 kHz 下是 16000 采样;切帧要按这个数出整秒的帧。"""
    assert samples_per_frame(1000) == 16000
    assert samples_per_frame(80) == 1280
    pcm = b"\x00" * (16000 * 4 * 3 + 100)  # 三秒零一点的 float32
    frames = slice_frames(pcm, samples_per_frame(1000))
    assert len(frames) == 3
    assert all(len(f) == 16000 * 4 for f in frames)


def test_session_stats_decision_bookkeeping() -> None:
    """决策延迟的分位数与超预算计数要按逐帧样本算,空样本给 None 而不是 0。"""
    s = SessionStats(index=0, scheduled_start_s=0.0, mode="audio")
    assert s.decision_p50_s is None and s.decision_p95_s is None and s.decision_max_s is None
    assert s.frames_over_budget(0.08) == 0
    s.decision_latencies.extend([0.02] * 90 + [0.5] * 10)
    assert s.decision_p50_s == pytest.approx(0.02)
    assert s.decision_p95_s == pytest.approx(0.5)
    assert s.decision_max_s == pytest.approx(0.5)
    assert s.frames_over_budget(0.08) == 10
    d = s.to_dict()
    assert d["decision_count"] == 100
    assert "decision_latencies" not in d, "原始样本不进 JSON"


def _decision_session(index: int, latencies, starving_s: float = 0.0):
    s = SessionStats(index=index, scheduled_start_s=0.0, mode="audio")
    s.end_reason = "lifetime_expired"
    s.connected_at = 0.0
    s.closed_at = 10.0
    s.decision_latencies.extend(latencies)
    s.listen_frames = len(latencies)
    s.underrun_total_s = starving_s
    return s


def test_summary_pools_decisions_and_passes_when_within_budget() -> None:
    """决策延迟按全部帧合并算分位数,不按会话平均;都在预算内且没断流时判 pass。"""
    from loadsim.arrival import ArrivalPlan
    from loadsim.runner import RunConfig, summarize

    sessions = [
        _decision_session(0, [0.5] * 100),
        _decision_session(1, [0.6] * 100),
    ]
    plan = ArrivalPlan(offsets=[0.0, 0.0], target_concurrency=2, duration_s=30.0, mode="ramp")
    cfg = RunConfig(frame_ms=1000)
    assert cfg.decision_budget_s == pytest.approx(1.0)
    summary = summarize(sessions, cfg, plan, sampled_concurrency=[2.0] * 10, peak_running=2)
    assert summary["decision_p95_s"] == pytest.approx(0.6)
    assert summary["decision_over_budget_ratio"] == 0.0
    assert summary["frames_answered_total"] == 200
    assert summary["listen_frames_total"] == 200
    assert summary["verdict"] == "pass"
    assert summary["verdict_reasons"] == []


def test_summary_fails_when_decision_p95_exceeds_budget_or_sessions_starve() -> None:
    """一个会话把 p95 拉过预算、另一个会话断流,判 fail 且两条原因都要点名。"""
    from loadsim.arrival import ArrivalPlan
    from loadsim.runner import RunConfig, summarize

    sessions = [
        _decision_session(0, [0.05] * 100),
        # 这 100 帧里 20 帧慢到 1.5s:合并后 200 帧中 10% 超预算,p95 落在 1.5
        _decision_session(1, [0.05] * 80 + [1.5] * 20, starving_s=2.0),
    ]
    plan = ArrivalPlan(offsets=[0.0, 0.0], target_concurrency=2, duration_s=30.0, mode="ramp")
    cfg = RunConfig(frame_ms=1000)
    summary = summarize(sessions, cfg, plan, sampled_concurrency=[2.0] * 10, peak_running=2)
    assert summary["decision_p95_s"] > 1.0
    assert summary["decision_over_budget_ratio"] == pytest.approx(0.10)
    assert summary["starving_session_ratio"] == 0.5
    assert summary["verdict"] == "fail"
    assert any("决策延迟" in r for r in summary["verdict_reasons"])
    assert any("断流" in r for r in summary["verdict_reasons"])
    assert not any("失败会话" in r for r in summary["verdict_reasons"])


def test_summary_without_decisions_is_not_pass() -> None:
    """没有任何带 response_id 的决策事件时不能判 pass:判据缺了一项,不是达标了。"""
    from loadsim.arrival import ArrivalPlan
    from loadsim.runner import RunConfig, summarize

    sessions = [_decision_session(0, [])]
    plan = ArrivalPlan(offsets=[0.0], target_concurrency=1, duration_s=30.0, mode="ramp")
    summary = summarize(sessions, RunConfig(), plan, sampled_concurrency=[1.0], peak_running=1)
    assert summary["decision_p95_s"] is None
    assert summary["verdict"] == "fail"
    assert summary["decision_over_budget_ratio"] == 0.0


def test_buffer_starts_listening_so_a_silent_session_never_starves() -> None:
    """模型从头听到尾、一句话没说的会话,不该被记成整段断流。

    会话刚建立时客户端手里没有任何音频,这不是「该播没播」,是「还没轮到播」。
    修之前初始状态是非倾听态,drain 从第一次 tick 就开始累加,一个 50 秒的纯倾听
    会话会被记出 50 秒断流、判成 100% 断流。
    """
    buf = PlaybackBuffer()
    buf.tick(1.0)
    buf.note_listen(now=1.0)
    buf.tick(50.0)
    buf.finish(50.0)
    assert buf.underrun_events == 0
    assert buf.underrun_total_s == 0.0


def test_listen_settles_the_debt_of_the_previous_turn() -> None:
    """模型决定听,就是上一轮说完了:欠下的播放债到此结清,不带进下一轮。

    时间线:0s 收到 1s 音频;3s 时已欠 2s;3s 收到 listen,断流在 3s 结束、记 2s;
    之后 10s 模型再开口给 1s 音频,这 1s 应当是新一轮的正常库存,能播到 11s,
    而不是先拿去填上一轮的 2s 窟窿。
    """
    buf = PlaybackBuffer()
    buf.note_audio(1.0, now=0.0)
    buf.tick(3.0)
    assert buf.underrun_events == 1
    buf.note_listen(now=3.0)
    assert buf.underrun_total_s == pytest.approx(2.0, abs=0.01)
    buf.note_audio(1.0, now=10.0)
    buf.tick(10.9)
    assert buf.underrun_events == 1, "新一轮的音频不该先去填旧账"
    buf.tick(11.5)
    assert buf.underrun_events == 2
    buf.finish(11.5)
    assert buf.underrun_total_s == pytest.approx(2.5, abs=0.02)
