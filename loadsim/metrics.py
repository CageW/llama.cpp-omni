"""单个 session 的观测指标。

这一层问的是「这个 session 体感好不好」,而不是「服务端总吞吐多少」。实时语音和
普通请求的根本区别在于:输出是一段连续的音频流,客户端要边收边播。服务端算得慢
一点、音频晚到一点,只要缓冲区里还有货就不会被听出来;一旦缓冲区见底,声音就断了。
所以这里的主指标不是平均延迟,而是播放缓冲见底的时长占比(starvation ratio),以及
延迟的分位数尾巴。

延迟的分母要小心:服务端发的是 80ms 一帧的音频,客户端每 20ms 播掉一帧的音频,
两者节奏不同,所以「相邻 delta 的间隔」本身不是延迟,只是到达节奏。真正能被听到的
是缓冲算术:收到一秒音频就多一秒库存,过了一秒墙钟就烧掉一秒库存。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# 服务端下行音频是 24 kHz 单声道 float32
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_BYTES_PER_SAMPLE = 4


def pcm_duration_s(n_bytes: int) -> float:
    return n_bytes / (OUTPUT_SAMPLE_RATE * OUTPUT_BYTES_PER_SAMPLE)


# 判定「库存是否见底」的容差。定时供料的会话里供给与消耗两个累加器会积累浮点尾差,
# 到几十帧后相差 1e-16 量级;不设容差会把健康会话记出大量长度为 0 的假断流事件。
# 1 微秒远小于任何有意义的音频时长,不会掩盖真实断流。
FLOAT_TOLERANCE_S = 1e-6


@dataclass
class SessionStats:
    """一个 session 的一生。字段名保持自明,方便直接进 JSON 和报表。"""

    index: int
    scheduled_start_s: float
    mode: str

    connect_started_at: float = 0.0
    connected_at: Optional[float] = None
    queue_done_at: Optional[float] = None
    init_sent_at: Optional[float] = None
    session_created_at: Optional[float] = None
    first_output_at: Optional[float] = None
    first_audio_at: Optional[float] = None
    closed_at: Optional[float] = None

    end_reason: str = "unknown"
    error: Optional[str] = None
    session_id: Optional[str] = None

    # 上行
    frames_sent: int = 0
    audio_sent_s: float = 0.0
    speech_sent_s: float = 0.0
    silence_frames_sent: int = 0
    send_errors: int = 0

    # 下行
    audio_deltas: int = 0
    text_deltas: int = 0
    audio_received_s: float = 0.0
    listen_events: int = 0

    # 播放缓冲
    underrun_total_s: float = 0.0
    underrun_events: int = 0
    max_underrun_s: float = 0.0
    worst_buffer_deficit_s: float = 0.0

    # 输出节奏
    audio_gap_samples: List[float] = field(default_factory=list)
    text_chars: int = 0

    queue_updates: int = 0

    # ---- 派生量 ----
    @property
    def session_duration_s(self) -> float:
        if self.connected_at is None or self.closed_at is None:
            return 0.0
        return self.closed_at - self.connected_at

    @property
    def setup_latency_s(self) -> Optional[float]:
        """从发起连接到拿到 session.created,包含排队等待。"""
        if self.connect_started_at is None or self.session_created_at is None:
            return None
        return self.session_created_at - self.connect_started_at

    @property
    def queue_wait_s(self) -> Optional[float]:
        """从发起到拿到首个排队反馈的时间,反映 worker 是否够用。"""
        if self.connect_started_at is None or self.queue_done_at is None:
            return None
        return self.queue_done_at - self.connect_started_at

    @property
    def init_latency_s(self) -> Optional[float]:
        """从 session.init 发出到 session.created,反映模型侧建会话的开销。"""
        if self.init_sent_at is None or self.session_created_at is None:
            return None
        return self.session_created_at - self.init_sent_at

    @property
    def ttfa_s(self) -> Optional[float]:
        """time to first audio:从会话建好到收到第一帧音频。"""
        if self.session_created_at is None or self.first_audio_at is None:
            return None
        return self.first_audio_at - self.session_created_at

    @property
    def speech_activity_ratio(self) -> float:
        """用户真正在说话的时间占上行总时长的比例。

        这个量可以和服务端侧的记录直接对照,是核对压测是否按预期形状发压的判据:
        工具声称按某组说话/静音参数发压,这一栏就该落在参数对应的活动比附近。
        电话语音的实测活动比是 32%~38%,明显低于直觉估计值。
        """
        if self.audio_sent_s <= 0:
            return 0.0
        return self.speech_sent_s / self.audio_sent_s

    @property
    def starve_ratio(self) -> float:
        """播放缓冲见底的时长占会话时长的比例。0 表示一次都没断过。"""
        dur = self.session_duration_s
        if dur <= 0:
            return 0.0
        return min(1.0, self.underrun_total_s / dur)

    @property
    def audio_gap_p50_s(self) -> Optional[float]:
        if not self.audio_gap_samples:
            return None
        return float(np.median(self.audio_gap_samples))

    @property
    def audio_gap_p99_s(self) -> Optional[float]:
        if not self.audio_gap_samples:
            return None
        return float(np.percentile(self.audio_gap_samples, 99))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "mode": self.mode,
            "scheduled_start_s": round(self.scheduled_start_s, 4),
            "end_reason": self.end_reason,
            "error": self.error,
            "session_id": self.session_id,
            "session_duration_s": round(self.session_duration_s, 4),
            "setup_latency_s": _r(self.setup_latency_s),
            "queue_wait_s": _r(self.queue_wait_s),
            "init_latency_s": _r(self.init_latency_s),
            "ttfa_s": _r(self.ttfa_s),
            "frames_sent": self.frames_sent,
            "audio_sent_s": round(self.audio_sent_s, 4),
            "speech_sent_s": round(self.speech_sent_s, 4),
            "silence_frames_sent": self.silence_frames_sent,
            "speech_activity_ratio": round(self.speech_activity_ratio, 4),
            "audio_deltas": self.audio_deltas,
            "text_deltas": self.text_deltas,
            "text_chars": self.text_chars,
            "audio_received_s": round(self.audio_received_s, 4),
            "listen_events": self.listen_events,
            "underrun_total_s": round(self.underrun_total_s, 4),
            "underrun_events": self.underrun_events,
            "max_underrun_s": round(self.max_underrun_s, 4),
            "starve_ratio": round(self.starve_ratio, 6),
            "audio_gap_p50_s": _r(self.audio_gap_p50_s),
            "audio_gap_p99_s": _r(self.audio_gap_p99_s),
            "send_errors": self.send_errors,
            "queue_updates": self.queue_updates,
        }


class PlaybackBuffer:
    """播放缓冲的库存模型。

    客户端边收边播,要回答的是压测里最要紧的那个问题:服务端产音频的速度有没有
    跟上实时。平均延迟好看但缓冲见底,对实时语音来说就是不能用。

    模型只做加法,不做比较判断:

        supply_s  累计收到的音频秒数
        drain_s   累计因为时间流过而消耗掉的秒数

    库存 = supply_s - drain_s。为负就是没货可播,负到多少就是已经断流多久。
    这两个量分别由「收到音频」和「时间推进」单调驱动,谁快谁慢一目了然,不存在
    「是否恰好等于零」这类在浮点下永远不可靠的判断。

    断流区间的两个端点用库存交叉的精确时刻解出:

        见底时刻 = 上次推进的时刻 + 推进前的库存   (库存被消耗完的瞬间)
        恢复时刻 = 到货时刻 - 到货后的库存          (余量从到货起才重新攒)

    压测客户端的调用节奏是 80ms(收到一帧)到 1s(空转采样),一帧音频迟到 200ms
    再赶上完全发生在两次调用之间。只看采样时刻,这段断流要么整段消失,要么多算
    一个采样间隔,而断流时长正是这套压测最核心的结论。

    这个模型是乐观估计:假设客户端一收到音频就能播,不做前置缓冲。真实客户端要
    多攒几百毫秒才起播,实际断流只会比这个数字更严重。

    双工模式的特殊之处:模型收到 kind=listen 就进入倾听状态,这时不该继续消耗
    库存(drain 暂停),否则会把「模型正常听、客户端正常等」算成断流。
    """

    supply_s: float = 0.0
    drain_s: float = 0.0
    last_at: Optional[float] = None
    underrun_total_s: float = 0.0
    underrun_events: int = 0
    max_underrun_s: float = 0.0
    _underrun_started: Optional[float] = None
    _is_listening: bool = False

    def note_audio(self, seconds: float, now: float) -> None:
        """到货:先把时间推进到 now,再加货。收到音频意味着模型在说话,退出倾听。"""
        self._advance(now)
        self.supply_s += seconds
        self._is_listening = False
        if self._underrun_started is not None and self.buffered_s > 0:
            self._close_gap(self._recovery_time())
            self._underrun_started = None

    def note_listen(self, now: float) -> None:
        """收到 kind=listen:模型进入倾听状态,暂停消耗库存。"""
        self._advance(now)
        self._is_listening = True

    def tick(self, now: float) -> None:
        self._advance(now)

    def finish(self, now: float) -> None:
        """会话结束时结算还挂着的那段断流。"""
        self._advance(now)
        if self._underrun_started is not None:
            self._close_gap(now)
            self._underrun_started = None

    @property
    def buffered_s(self) -> float:
        """当前囤着多少秒音频可播,不会为负。"""
        return max(0.0, self.supply_s - self.drain_s)

    @property
    def starving(self) -> bool:
        return self._underrun_started is not None

    # ---- 内部 ----
    def _advance(self, now: float) -> None:
        """把墙钟推进到 now:消耗掉这段时间,并在库存转负时记下断流起点。

        倾听状态下不消耗库存:模型说完了在等用户说话,这是双工协议的正常状态,
        不能算成断流。

        这里用一个极小的容差(1 微秒)判定「是否真的见底」。定时供料的会话里,
        供给和消耗两个累加器各自积累浮点尾差,到几十帧之后二者会相差 1e-16 量级;
        如果不加容差,每个健康的会话都会被记出若干个长度为零的假断流事件,而累计
        断流时长仍是 0——报告上就会出现「从没断过,但断了一百次」这种自相矛盾。
        容差取 1 微秒,远小于任何有意义的音频时长(一帧 80ms),不会掩盖真实断流。
        """
        if self.last_at is None:
            self.last_at = now
            return
        elapsed = now - self.last_at
        if elapsed <= 0:
            return
        level_before = self.supply_s - self.drain_s
        # 倾听状态下暂停消耗
        if not self._is_listening:
            self.drain_s += elapsed
        self.last_at = now
        if self._underrun_started is not None:
            return
        if self.supply_s - self.drain_s >= -FLOAT_TOLERANCE_S:
            return
        if level_before > FLOAT_TOLERANCE_S:
            # 库存正好在这一区间内被消耗完,起点 = 区间起点 + 耗尽前的库存
            self._underrun_started = now - elapsed + level_before
        else:
            # 本来就没货,断流从这一区间的起点就成立
            self._underrun_started = now - elapsed
        self.underrun_events += 1

    def _close_gap(self, end: float) -> None:
        if self._underrun_started is None:
            return
        gap = max(0.0, end - self._underrun_started)
        self.underrun_total_s += gap
        self.max_underrun_s = max(self.max_underrun_s, gap)

    def _recovery_time(self) -> float:
        """库存回到正数的那一刻,即解 supply_s == 累计消耗 的时刻。

        这一帧覆盖的是它到达之前的那段时间,所以到货之后多出来的余量要往回退:
        余量为 b 表示库存在到货时刻之前 b 秒就已经转正了。只有到货正好把欠账
        还平(余量为零)时,恢复时刻才等于到货时刻本身。
        """
        if self.last_at is None:
            return 0.0
        return self.last_at - self.buffered_s


def _r(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(value, 4)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(values, q))
