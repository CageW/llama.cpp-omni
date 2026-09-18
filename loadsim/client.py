"""一个压测客户端(会话)。

它的行为刻意保持朴素:连上、排队、建会话、持续上行音频帧、持续收下行事件、按需要
关闭。不做播放、不做重采样、不管音频内容——因为它要回答的问题是服务端能不能扛住
这么多并发会话,不是模型答得好不好。上行音频固定复用同一段素材,保证不同 session
之间只有「什么时候上线、说话节奏如何」这两个变量在变。

协议细节以 MiniCPM-o-Demo 仓库的实时协议文档为准:客户端要等到 session.queue_done
才能发 session.init;音频是 16 kHz 单声道 float32,base64 编码;音频双工没有
response.done,输出边界靠 kind=listen 表达。服务端给第 k 个 input.append 的响应
命名为 <session_id>_resp_<k>(k 从 1 起),listen/text 两种 delta 各带这个 response_id,
客户端靠它把「第 k 帧发出」与「模型对第 k 帧的决策到达」配成一对,算出决策延迟。
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import ssl
import time
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from loadsim.activity import FRAME_MS, SpeechActivity, SpeechPattern
from loadsim.metrics import PlaybackBuffer, SessionStats, pcm_duration_s
from loadsim.video import VideoSource

# 播放缓冲按墙钟推进的最长间隔。它原本与帧长绑在一起,但帧长改成 1000ms 后,
# 收不到消息时一秒才推进一次缓冲,断流起止时刻的分辨率会退化到一秒,所以单独封顶。
PLAYBACK_TICK_MAX_S = 0.08

_RESP_INDEX = re.compile(r"_resp_(\d+)$")


def build_ssl_context(url: str, insecure: bool, ca_file: Optional[str]) -> Optional[ssl.SSLContext]:
    """按 URL 与参数决定 wss 客户端用什么 TLS 上下文。

    三种情况要分开处理,混在一起会让压测在 TLS 握手阶段就全军覆没,而报出来的错
    是「连接失败」,看不出根因在证书上:

    - 目标是 wss 且给了 CA 文件:正常校验,这是生产环境该用的方式;
    - 目标是 wss 且没给 CA:用系统信任库。如果服务用的是自签证书,这里会失败,
      应当改用 --ca-file 而不是直接跳过校验;
    - --insecure:不校验。只在对内网的临时验证里用,压测报告里要把这一点写明,
      否则数字的适用范围会被误读。
    """
    if not url.startswith("wss://"):
        return None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    if ca_file:
        return ssl.create_default_context(cafile=ca_file)
    return ssl.create_default_context()


class SimClient:
    """一个模拟会话。每次 run() 走完从连接到关闭的完整生命周期。"""

    def __init__(
        self,
        *,
        index: int,
        url: str,
        mode: str,
        frames: list[bytes],
        speech: SpeechPattern,
        lifetime_s: float,
        scheduled_start_s: float,
        init_payload: Optional[Dict[str, Any]] = None,
        max_size: int = 64 * 1024 * 1024,
        open_timeout_s: float = 30.0,
        receive_idle_timeout_s: float = 60.0,
        send_silence_during_pause: bool = True,
        ssl_context: Optional[ssl.SSLContext] = None,
        handshake_timeout_s: float = 30.0,
        frame_ms: int = FRAME_MS,
        video: Optional["VideoSource"] = None,
    ) -> None:
        self.index = index
        self.url = url
        self.mode = mode
        self.frames = frames
        self.frame_s = frame_ms / 1000.0
        self.speech = speech
        self.lifetime_s = lifetime_s
        self.stats = SessionStats(index=index, scheduled_start_s=scheduled_start_s, mode=mode)
        self.buffer = PlaybackBuffer()
        self.init_payload = init_payload or {"system_prompt": "You are a helpful assistant."}
        self.max_size = max_size
        self.open_timeout_s = open_timeout_s
        self.receive_idle_timeout_s = receive_idle_timeout_s
        self.send_silence_during_pause = send_silence_during_pause
        self.ssl_context = ssl_context
        self.handshake_timeout_s = handshake_timeout_s
        self.video = video

        self._frame_idx = 0
        self._last_audio_at: Optional[float] = None
        self._closed = asyncio.Event()
        self._last_recv_at = 0.0
        self._activity: Optional[SpeechActivity] = None
        self._send_cursor = 0.0
        # 第 k 个 input.append 的发出时刻,收到对应决策后即删,所以长度不超过在途帧数
        self._sent_at: Dict[int, float] = {}

    # ------------------------------------------------------------------ 上行
    async def _pump_input(self, ws) -> None:
        """按帧长节奏上行音频帧。

        send_silence_during_pause 决定静音段的行为,两种口径都有真实对应物:

        - True(默认):静音段照常上行,只是内容是静音帧。真实浏览器/原生客户端
          很少停止上行,麦克风采集是连续的,不发的只是没声音。这是服务端实际会
          收到的流量形状。
        - False:静音段完全不上行。这是移动端省流量时的做法,会让服务端的 TDM
          时间片上出现空档,和上一种口径的负载明显不同。

        发送时刻加入抖动,避免所有客户端锁步到同一个 tick。

        节拍按「从开始算起第几帧」定,不是按「上一帧发完之后再等 80ms」定。两者
        在发送不落后时没有区别,但一旦某次 send 被阻塞(服务端读得慢、本地事件
        循环忙),前者会让后续几帧紧跟着补发出去,把客户端重新拉回实时节拍;后者
        则会让落后逐次累积,一个 5 分钟的会话最后会比实时慢出好几秒,上行总量也
        就不再等于音频时长。补发意味着客户端不会因为服务端慢而自动降载,这正是
        压测要的:压力应当真实地传导到服务端,而不是被客户端悄悄吸收。
        """
        started = time.monotonic()
        self._activity = SpeechActivity(self.speech)
        silence_frame = b"\x00" * len(self.frames[0])
        while not self._closed.is_set():
            now = time.monotonic()
            if now - started >= self.lifetime_s:
                return
            talking = self._activity.advance(self.frame_s)
            target = started + self._send_cursor
            self._send_cursor += self.frame_s
            if talking or self.send_silence_during_pause:
                if talking:
                    frame = self.frames[self._frame_idx % len(self.frames)]
                    self._frame_idx += 1
                else:
                    frame = silence_frame
                    if self.send_silence_during_pause:
                        # 静音帧要单独记:它同样占用上行带宽与服务端的处理时间,
                        # 但不是「用户说的话」。两者混在一个字段里,读报告的人会
                        # 以为上行音频时长等于用户说话时长,进而把语音活动比算错。
                        self.stats.silence_frames_sent += 1
                inp: Dict[str, Any] = {"audio": base64.b64encode(frame).decode("ascii")}
                video_frames = self.video.next_frames() if self.video is not None else []
                if video_frames:
                    # 字段名与产品前端一致(realtime-api/video.md):input.video_frames 是
                    # base64 JPEG 列表,服务端只取第一帧;max_slice_nums 是本次输入的切片数。
                    inp["video_frames"] = video_frames
                    inp["max_slice_nums"] = self.video.max_slice_nums
                payload = {"type": "input.append", "input": inp}
                try:
                    await ws.send(json.dumps(payload))
                except ConnectionClosed:
                    return
                except Exception:
                    self.stats.send_errors += 1
                else:
                    self.stats.frames_sent += 1
                    self.stats.video_frames_sent += len(video_frames)
                    self._sent_at[self.stats.frames_sent] = time.monotonic()
                    self.stats.audio_sent_s += self.frame_s
                    if talking:
                        self.stats.speech_sent_s += self.frame_s
            sleep_for = target + self.speech.sample_jitter_s() - time.monotonic()
            if sleep_for > 0:
                try:
                    await asyncio.wait_for(self._closed.wait(), timeout=sleep_for)
                    return
                except asyncio.TimeoutError:
                    pass

    # ------------------------------------------------------------------ 下行
    def _handle_event(self, msg: Dict[str, Any], now: float) -> None:
        t = msg.get("type")
        if t == "session.queue_update":
            self.stats.queue_updates += 1
        elif t == "response.output.delta":
            kind = msg.get("kind")
            if kind == "audio":
                b64 = msg.get("audio") or ""
                n_bytes = base64.b64decode(b64, validate=False).__len__() if b64 else 0
                if n_bytes:
                    self.buffer.note_audio(pcm_duration_s(n_bytes), now)
                    self.stats.audio_deltas += 1
                    self.stats.audio_received_s += pcm_duration_s(n_bytes)
                    if self.stats.first_output_at is None:
                        self.stats.first_output_at = now
                    if self.stats.first_audio_at is None:
                        self.stats.first_audio_at = now
                    if self._last_audio_at is not None:
                        self.stats.audio_gap_samples.append(now - self._last_audio_at)
                    self._last_audio_at = now
            elif kind == "text":
                self.stats.text_deltas += 1
                self.stats.text_chars += len(msg.get("text") or "")
                if self.stats.first_output_at is None:
                    self.stats.first_output_at = now
                self._resolve_decision(msg, kind, now)
            elif kind == "listen":
                self.buffer.note_listen(now)
                self.stats.listen_events += 1
                self._resolve_decision(msg, kind, now)

    def _resolve_decision(self, msg: Dict[str, Any], kind: str, now: float) -> None:
        """把一条 listen/text delta 归到它回应的那一帧,记下决策延迟。

        只认每帧的第一条:模型说话时一帧会带出多条 text delta,后面的都是同一次决策
        的续流,不是新的决策。audio delta 不参与,它是 TTS 异步吐的,与决策时刻无关。
        metrics 里的 prefill_ms / generate_ms 是服务端自报的本帧耗时,单独收着,用来
        把「决策慢」拆成「排队慢」还是「算得慢」。
        """
        m = _RESP_INDEX.search(msg.get("response_id") or "")
        if not m:
            return
        sent = self._sent_at.pop(int(m.group(1)), None)
        if sent is None:
            return
        self.stats.decision_latencies.append(now - sent)
        if kind == "text":
            self.stats.speak_frames += 1
        else:
            self.stats.listen_frames += 1
        metrics = msg.get("metrics")
        if isinstance(metrics, dict):
            if metrics.get("generate_ms") is not None:
                self.stats.server_decode_ms.append(float(metrics["generate_ms"]))
            if metrics.get("prefill_ms") is not None:
                self.stats.server_prefill_ms.append(float(metrics["prefill_ms"]))
            # 服务端自报「这一帧编了几个视觉切片」。声称发了视频却全程为零,说明视频那一路
            # 根本没生效,这时测出来的是纯音频负载,必须让它暴露出来而不是记成一次正常压测。
            if metrics.get("vision_slices"):
                self.stats.vision_slices_total += int(metrics["vision_slices"])
            if metrics.get("vision_tokens"):
                self.stats.vision_tokens_total += int(metrics["vision_tokens"])

    async def _pump_output(self, ws) -> None:
        """收下行事件,并按墙钟推进播放缓冲的消耗。"""
        while not self._closed.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(self.frame_s, PLAYBACK_TICK_MAX_S))
            except asyncio.TimeoutError:
                self.buffer.tick(time.monotonic())
                if self.stats.connected_at is not None and (
                    time.monotonic() - self._last_recv_at > self.receive_idle_timeout_s
                ):
                    self.stats.error = f"控制面静默超过 {self.receive_idle_timeout_s:.0f}s"
                    return
                continue
            except ConnectionClosed:
                return
            now = time.monotonic()
            self._last_recv_at = now
            if isinstance(raw, bytes):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            self._handle_event(msg, now)
            self.buffer.tick(now)

    # ------------------------------------------------------------------ 生命周期
    async def run(self) -> SessionStats:
        self.stats.connect_started_at = time.monotonic()
        self._last_recv_at = self.stats.connect_started_at
        try:
            connect_kwargs: Dict[str, Any] = {
                "max_size": self.max_size,
                "open_timeout": self.open_timeout_s,
            }
            if self.ssl_context is not None:
                connect_kwargs["ssl"] = self.ssl_context
            async with websockets.connect(self.url, **connect_kwargs) as ws:
                self.stats.connected_at = time.monotonic()
                if not await self._handshake(ws):
                    return self.stats
                send_task = asyncio.create_task(self._pump_input(ws))
                recv_task = asyncio.create_task(self._pump_output(ws))
                done, pending = await asyncio.wait(
                    {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                await self._close(ws)
        except asyncio.TimeoutError:
            self.stats.end_reason = "connect_timeout"
            self.stats.error = "连接或握手超时"
        except Exception as exc:  # noqa: BLE001 - 压测里任何异常都要记到这一条会话上,不能炸整个 run
            self.stats.end_reason = "error"
            self.stats.error = f"{type(exc).__name__}: {exc}"
        finally:
            now = time.monotonic()
            self.buffer.finish(now)
            self.stats.closed_at = now
            self.stats.underrun_total_s = self.buffer.underrun_total_s
            self.stats.underrun_events = self.buffer.underrun_events
            self.stats.max_underrun_s = self.buffer.max_underrun_s
            self.stats.frames_unanswered = len(self._sent_at)
            # 断流区间改成相对会话建立时刻的秒数,读起来直接是「第几秒断到第几秒」
            t_ref = self.stats.session_created_at or self.stats.connected_at or now
            self.stats.underrun_intervals = [
                (round(a - t_ref, 2), round(b - t_ref, 2)) for a, b in self.buffer.underrun_intervals
            ]
            if self.stats.end_reason == "unknown":
                self.stats.end_reason = "lifetime_expired"
        return self.stats

    async def _handshake(self, ws) -> bool:
        """等到 queue_done 再发 session.init,然后等 session.created。

        等待上限是 handshake_timeout_s,它同时覆盖「排队等 worker」和「排到之后建会话」
        两段。默认值和 websockets 的连接超时一致,但容量压测时应当显式调大:worker
        全忙时排队几十秒是正常现象,用默认值会把「在排队」误报成「服务无响应」。
        """
        deadline = time.monotonic() + self.handshake_timeout_s
        init_sent = False
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                waited = time.monotonic() - self.stats.connect_started_at
                if self.stats.queue_done_at is None:
                    # 一直没等到排队结束。这不是故障,是容量不够:worker 全忙,
                    # 新会话卡在 gateway 队列里。报成「握手超时」会把容量问题说成
                    # 服务故障,读报告的人会去查日志找崩溃,而真正该做的是加 worker。
                    self.stats.end_reason = "queue_timeout"
                    self.stats.error = (
                        f"在队列里等了 {waited:.1f}s 仍未排到 worker(worker 全忙);"
                        f"排队等待计入 setup_latency,本条不算服务端故障"
                    )
                else:
                    self.stats.end_reason = "init_timeout"
                    self.stats.error = (
                        f"排到 worker 后 {waited - (self.stats.queue_wait_s or 0):.1f}s 内没收到 session.created"
                    )
                return False
            except ConnectionClosed as exc:
                self.stats.end_reason = "closed_during_handshake"
                self.stats.error = str(exc)
                return False
            now = time.monotonic()
            if isinstance(raw, bytes):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("type")
            if t in ("session.queued", "session.queue_update"):
                self.stats.queue_updates += 1
                self._last_recv_at = now
                continue
            if t in ("session.queue_done", "queue_done"):
                self.stats.queue_done_at = self.stats.queue_done_at or now
                self._last_recv_at = now
                if not init_sent:
                    self.stats.init_sent_at = time.monotonic()
                    payload = {"type": "session.init", "payload": self.init_payload}
                    await ws.send(json.dumps(payload))
                    init_sent = True
                continue
            if t == "session.created":
                self.stats.session_created_at = now
                self.stats.session_id = msg.get("session_id")  # type: ignore[attr-defined]
                self._last_recv_at = now
                return True
            if t == "error":
                self.stats.end_reason = "server_error"
                self.stats.error = json.dumps(msg, ensure_ascii=False)[:400]
                return False
        return False

    async def _close(self, ws) -> None:
        self._closed.set()
        self.stats.end_reason = "lifetime_expired"
        try:
            await ws.send(json.dumps({"type": "session.close", "reason": "client_stop"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(raw) if isinstance(raw, str) else {}
            if msg.get("type") == "session.closed":
                self.stats.end_reason = f"closed:{msg.get('reason', 'unknown')}"
        except Exception:
            pass
