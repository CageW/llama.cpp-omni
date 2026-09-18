"""补位时刻必须只由 --seed 决定。

策略 A/B 比较的前提是两臂看到同一份到达序列。补位延迟若来自全局 random,
同一份配置跑两遍的补位时刻就不同,测出来的策略差异里混着到达序列的差异。
"""
import asyncio
import random

from loadsim.fake_server import serve_fake
from loadsim.runner import RunConfig, run_load


def _launch_offsets(seed: int) -> list:
    async def go():
        _server, port, stop = await serve_fake()
        try:
            cfg = RunConfig(
                url=f"ws://127.0.0.1:{port}/v1/realtime?mode=audio",
                arrival_mode="poisson", target_concurrency=3, duration_s=8.0,
                ramp_s=1.0, reuse=True, retry_mean_s=0.4,
                lifetime_mean_s=1.5, lifetime_dist="lognormal",
                lifetime_min_s=0.5, lifetime_max_s=3.0,
                seed=seed, frame_ms=1000,
            )
            rep = await run_load(cfg, frames=[b"\x00" * 32000])
            return sorted(s["scheduled_start_s"] for s in rep.sessions)
        finally:
            await stop()

    return asyncio.run(go())


def test_refill_offsets_reproducible_under_same_seed():
    random.seed(1)            # 外界扰动全局 random 不该改变结果
    a = _launch_offsets(31)
    random.seed(999)
    b = _launch_offsets(31)
    assert len(a) > 3, "补位没发生,这个用例就没验到东西"
    assert len(a) == len(b), (len(a), len(b))
    # 补位由真实退出事件驱动,墙钟上有调度抖动;比到 50ms 量级即可
    for x, y in zip(a, b):
        assert abs(x - y) < 0.35, (a, b)


def test_refill_holds_target_concurrency():
    """补位要把并发顶到目标附近。

    退出速率是目标并发除以会话时长;补位的等待若在投放循环里串行执行,每秒最多补
    1/retry_mean 个会话,目标一高就补不上,实际并发会停在一个与目标无关的值上,而
    报告里仍写着目标并发。这里把两个速率拉开一个数量级:目标 12 路、会话 2 秒、
    重连间隔 1 秒,需要 6 个每秒,串行补位最多 1 个每秒。
    """
    async def go():
        _server, port, stop = await serve_fake()
        try:
            cfg = RunConfig(
                url=f"ws://127.0.0.1:{port}/v1/realtime?mode=audio",
                arrival_mode="poisson", target_concurrency=12, duration_s=22.0,
                ramp_s=2.0, reuse=True, retry_mean_s=1.0,
                lifetime_mean_s=2.0, lifetime_dist="fixed",
                lifetime_min_s=0.5, lifetime_max_s=4.0,
                seed=5, frame_ms=1000,
            )
            return await run_load(cfg, frames=[b"\x00" * 32000])
        finally:
            await stop()

    rep = asyncio.run(go())
    steady = rep.summary["concurrency_steady_mean"]
    assert steady >= 12 * 0.7, "稳态并发只有 %s,补位没跟上" % steady
