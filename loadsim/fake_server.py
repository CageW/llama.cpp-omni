"""一个假服务端,用来在没有 GPU 的机器上验证压测工具本身。

它只做一件事:复现实时双工协议的外部时序(排队 → session.created → 持续上行 →
持续下行 listen/text/audio),并故意让每会话的合成速度随并发下降。这样工具算出来的
并发曲线、播放缓冲断流、延迟分位数就都能被检验出对错——如果工具的数学错了,这里
就能看出来,不必每次都上一台真机器。

它不复现模型的任何语义:不推理、不看音频内容,收到上行就直接按固定节奏回下行。
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Callable, Dict, Set, Tuple

import websockets

# 假服务端每次下行吐出的音频时长与真实服务端同量级
DOWNLINK_CHUNK_S = 0.08
# 单会话的合成速度:每秒墙钟能产出多少秒音频。1.0 表示实时,<1 表示会跟不上
BASE_REALTIME_FACTOR = 1.6
# 每多一个并发会话,速度下降的系数。用来制造「并发升高后开始断流」的观感
CONCURRENCY_DECAY = 0.16
# 建会话的固定开销,用来制造建会话延迟随并发上升
INIT_BASE_S = 0.15
INIT_PER_SESSION_S = 0.03


class FakeServerState:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.total = 0
        self.sessions: Set[str] = set()
        self.lock = asyncio.Lock()

    def realtime_factor(self, concurrency: int) -> float:
        """并发越高,单会话分到的合成速度越低。"""
        return max(0.05, BASE_REALTIME_FACTOR / (1.0 + CONCURRENCY_DECAY * max(0, concurrency - 1)))


async def _handle(ws, state: FakeServerState, path: str = "", init_cost: bool = True,
                  echo_input: bool = False, report_vision: bool = True) -> None:
    sid = f"fake_{state.total + 1}_{int(time.time() * 1000)}"
    active = 0
    tasks: list[asyncio.Task] = []
    try:
        await ws.send(json.dumps({"type": "session.queued", "session_id": sid, "position": 0}))
        await ws.send(json.dumps({"type": "session.queue_update", "session_id": sid, "eta_s": 1.0}))
        await ws.send(json.dumps({"type": "session.queue_done", "session_id": sid}))

        init_payload: Dict[str, Any] = {}
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "session.init":
                init_payload = msg.get("payload") or {}
                break
            if msg.get("type") == "session.close":
                return

        async with state.lock:
            state.active += 1
            state.total += 1
            state.peak = max(state.peak, state.active)
            active = state.active

        if init_cost:
            await asyncio.sleep(INIT_BASE_S + INIT_PER_SESSION_S * active)
        await ws.send(json.dumps({"type": "session.created", "session_id": sid, "mode": "full_duplex"}))
        _ = init_payload

        stop = asyncio.Event()

        async def downlink() -> None:
            """持续吐出合成音频,节奏随当时并发变化。

            假服务端不模拟双工的 SPEAK/LISTEN 切换,只持续吐音频,因为:
            1. 它要验证的是压测工具的到达模型、播放缓冲算术、并发采样,
               这些都不依赖 listen 事件;
            2. listen 的语义处理已由单元测试覆盖,假服务端加 listen 只会让
               「健康会话是否误判断流」这个最基础的校验变复杂。

            音频内容只是有内容的字节(不全零),这样客户端侧的字节数校验不会
            因为压缩或全零优化而失真。
            """
            delta_samples = int(24000 * DOWNLINK_CHUNK_S)
            payload_bytes = bytes((i * 37 + 11) % 251 for i in range(delta_samples * 4))
            while not stop.is_set():
                factor = state.realtime_factor(state.active)
                await asyncio.sleep(DOWNLINK_CHUNK_S / factor)
                await ws.send(
                    json.dumps(
                        {
                            "type": "response.output.delta",
                            "kind": "audio",
                            "audio": base64.b64encode(payload_bytes).decode("ascii"),
                            "response_id": "resp_fake",
                            "session_id": sid,
                        }
                    )
                )

        async def uplink() -> None:
            """收上行包。

            echo_input=True 时对每个 input.append 回一条 listen delta,response_id 带上
            与产品一致的 `_resp_<k>` 后缀,并在 metrics 里报这一帧编了几个视觉切片。
            这是验证「声称发视频、服务端却没处理」这条门禁所必需的:门禁判的正是
            metrics.vision_slices,没有它就只能测到客户端发出去了、测不到服务端收没收到。
            report_vision=False 用来扮演一个把 video_frames 丢掉的服务端。
            """
            seq = 0
            while not stop.is_set():
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "session.close":
                    stop.set()
                    return
                if echo_input and msg.get("type") == "input.append":
                    seq += 1
                    inp = msg.get("input") or {}
                    frames = inp.get("video_frames") or []
                    slices = 0
                    if frames and report_vision:
                        slices = int(inp.get("max_slice_nums") or 1)
                    await ws.send(json.dumps({
                        "type": "response.output.delta",
                        "kind": "listen",
                        "session_id": sid,
                        "response_id": f"{sid}_resp_{seq}",
                        "metrics": {"vision_slices": slices, "vision_tokens": slices * 64},
                    }))

        tasks = [asyncio.create_task(downlink()), asyncio.create_task(uplink())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        try:
            await ws.send(json.dumps({"type": "session.closed", "reason": "client_stop"}))
        except Exception:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        for t in tasks:
            t.cancel()
        if active:
            async with state.lock:
                state.active -= 1


async def serve_fake(
    host: str = "127.0.0.1", port: int = 0,
    echo_input: bool = False, report_vision: bool = True,
) -> Tuple[Any, int, Callable[[], Any]]:
    """起假服务端,返回 (server, 实际端口, 关闭函数)。传 port=0 由系统分配端口。

    echo_input: 对每个 input.append 回一条带 response_id 与 metrics 的 listen delta。
    report_vision: echo_input 时,收到视频帧要不要在 metrics 里报视觉切片。设 False 就是
        扮演一个把 video_frames 丢掉的服务端,用来验证压测工具能不能发现这件事。
    """
    state = FakeServerState()
    server = await websockets.serve(
        lambda ws, path=None: _handle(ws, state, path or "", echo_input=echo_input,
                                      report_vision=report_vision),
        host, port)
    sock = next(iter(server.sockets))
    actual_port = sock.getsockname()[1]

    async def stop() -> None:
        server.close()
        await server.wait_closed()

    return server, actual_port, stop


async def serve_unthrottled(
    host: str = "127.0.0.1", port: int = 0
) -> Tuple[Any, int, Callable[[], Any]]:
    """起一个不做任何限速的桩,用于发压机自证。

    与 serve_fake 的区别:它不随并发放慢下行节奏,也不加建会话开销。客户端在这上面
    跑出的断流只可能来自发压机自己(CPU 跑不动、事件循环被阻塞、本地网络排队),
    所以它是区分「服务端慢」和「发压机慢」的对照。
    """
    state = FakeServerState()
    state.realtime_factor = lambda concurrency: BASE_REALTIME_FACTOR  # type: ignore[assignment]
    server = await websockets.serve(
        lambda ws, path=None: _handle(ws, state, path or "", init_cost=False), host, port
    )
    sock = next(iter(server.sockets))
    actual_port = sock.getsockname()[1]

    async def stop() -> None:
        server.close()
        await server.wait_closed()

    return server, actual_port, stop


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="假实时双工服务端,仅用于压测工具自检")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8006)
    args = ap.parse_args()

    async def _run() -> None:
        server, actual_port, _ = await serve_fake(args.host, args.port)
        print(f"假服务端监听 ws://{args.host}:{actual_port}/v1/realtime?mode=audio, Ctrl-C 退出")
        await server.wait_closed()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
