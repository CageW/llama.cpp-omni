"""确定性后端:用来在没有 GPU 的机器上跑通真实的 gateway + worker 链路。

它不是模拟器,是**替身**。用途是让 omni-perf-op/loadsim 的压测工具面对一个真实的
服务端代码路径:真的 gateway.py 做排队与 worker 分配,真的 worker.py 做会话桥接与
状态机,只有最里面一层「模型算多久、吐多少音频」换成固定公式。

这样做的价值在于:压测工具最容易错的地方是协议与生命周期(什么时候该等
session.queue_done、init 之后多久才会来 session.created、音频 delta 该按什么节奏
吐),而这些全在真 gateway 与真 worker 里,是可以真实校验的。替换掉的只有模型本身,
而模型算多快本来就是要压测去测的量,不是校验工具正确性的基准。

协议面照 py_backend/server.py 实现:客户端(worker)连 /backend,先发 session.init,
之后只发 input.append;关闭走 HTTP POST /sessions/{id}/close。服务端吐
session.created、response.output.delta(kind=listen/text/audio)。

吐音频的节奏用一个简单的并发模型:每个会话的基础产出速率固定,并发越高越慢,
慢到低于实时就会在客户端侧表现为播放缓冲见底。这个公式是刻意写死的,所以
「同一配置重跑结果一致」是可验证的。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("stub_backend")

OUTPUT_SAMPLE_RATE = 24000
DELTA_MS = 80
# 单会话在零并发时的产出速率(音频秒 / 墙钟秒)。> 1 表示吐得比实时快。
BASE_RATE = 1.35
# 每多一个并发会话,速率按这个比例衰减
CONCURRENCY_DECAY = 0.11
# init(建会话)的固定开销,再加一点随并发增长的部分
INIT_BASE_S = 0.10
INIT_PER_SESSION_S = 0.02

_state: dict = {"active": 0, "total": 0, "peak": 0}


def _rate_for(concurrency: int) -> float:
    return max(0.02, BASE_RATE / (1.0 + CONCURRENCY_DECAY * max(0, concurrency - 1)))


def _pcm_delta_bytes() -> bytes:
    samples = OUTPUT_SAMPLE_RATE * DELTA_MS // 1000
    # 有内容的字节,避免全零被压缩或优化
    return bytes((i * 37 + 11) % 251 for i in range(samples * 4))


DELTA_BYTES = _pcm_delta_bytes()
DELTA_B64 = base64.b64encode(DELTA_BYTES).decode("ascii")

app = FastAPI(title="omni stub backend")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend": "stub", "active": _state["active"], "peak": _state["peak"]}


@app.post("/sessions/{session_id}/close")
async def close_session(session_id: str, request: Request) -> dict:
    sessions = app.state.sessions
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    reason = "client_closed"
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("reason"):
            reason = str(body["reason"])
    except Exception:
        pass
    session["close"].set()
    return {"ok": True, "session_id": session_id, "closed": True, "reason": reason}


@app.websocket("/backend")
async def backend_ws(ws: WebSocket) -> None:
    await ws.accept()
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    close_event = asyncio.Event()
    app.state.sessions[session_id] = {"close": close_event}
    active = 0
    downlink: asyncio.Task | None = None
    uplink: asyncio.Task | None = None

    async def send(event_type: str, **fields) -> None:
        data = {"type": event_type, **{k: v for k, v in fields.items() if v is not None}}
        data["server_send_ts"] = time.time()
        await ws.send_json(data)

    async def downlink_loop() -> None:
        """按当前并发对应的速率持续吐音频 delta。"""
        while not close_event.is_set():
            rate = _rate_for(_state["active"])
            await asyncio.sleep((DELTA_MS / 1000.0) / rate)
            await send(
                "response.output.delta",
                kind="audio",
                audio=DELTA_B64,
                response_id=f"resp_{session_id}",
                session_id=session_id,
            )

    async def uplink_loop() -> None:
        while not close_event.is_set():
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                close_event.set()
                return
            msg = json.loads(raw)
            if msg.get("type") == "input.append":
                continue
            close_event.set()
            return

    try:
        first = json.loads(await ws.receive_text())
        if first.get("type") != "session.init":
            await ws.close(code=1011, reason="first message must be session.init")
            return
        params = first.get("payload") if isinstance(first.get("payload"), dict) else {}
        mode = str(params.get("mode") or "full_duplex")

        _state["active"] += 1
        _state["total"] += 1
        _state["peak"] = max(_state["peak"], _state["active"])
        active = _state["active"]

        # 建会话开销:固定部分 + 随并发增长的部分,用来让排队/容量问题能被观测到
        await asyncio.sleep(INIT_BASE_S + INIT_PER_SESSION_S * active)

        await send(
            "session.created",
            session_id=session_id,
            mode=mode,
            metrics={"init_active_sessions": active},
        )

        downlink = asyncio.create_task(downlink_loop())
        uplink = asyncio.create_task(uplink_loop())
        done, pending = await asyncio.wait(
            {downlink, uplink, asyncio.create_task(close_event.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        await send("session.closed", session_id=session_id, reason="client_closed")
    except WebSocketDisconnect:
        logger.info("session %s disconnected", session_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("session %s failed: %s", session_id, exc)
    finally:
        close_event.set()
        for t in (downlink, uplink):
            if t is not None:
                t.cancel()
        if active:
            _state["active"] -= 1
        app.state.sessions.pop(session_id, None)


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="确定性后端,替代 py_backend/server.py 跑协议链路")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=22500)
    args = ap.parse_args()

    app.state.sessions = {}
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
