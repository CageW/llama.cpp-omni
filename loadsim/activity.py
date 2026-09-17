"""单个客户端在会话里干什么：说话、停顿、被打断。

到达分布决定「什么时候来」,这个模块决定「来了之后怎么用」。两者都会显著改变服务端
负载,但改变方式不同:到达分布改变瞬时并发的形状,说话活动改变每一时刻有多少个会话
真的在提交计算。

一个纯双工会话里客户端做的事情是:持续上行音频帧。上行本身不区分「用户正在说话」和
「用户没说话」——服务端看到的是连续流。但真实用户是有说话的段落和停顿的,而模型在
停顿处的行为(继续 LISTEN)和在有语音输入处的行为(可能 SPEAK)完全不同。所以这里
按「说话段 (talk spurt) + 静音间隔 (pause)」来编排上行内容,并额外支持两种把活动
加剧的模式:

- barge_in: 静音间隔缩短到接近零,模拟用户抢话。上行变得几乎连续,服务端侧就是
  一直有新输入要处理。
- 说话段内抖动: 每帧的实际发送时刻加一点抖动,模拟真实网络与浏览器音频采集的抖动,
  避免所有客户端精确同步到同一个 tick。

帧长默认 80ms(1280 采样 @16kHz),这是 OpenAI Realtime 默认 server VAD 的分块粒度。
但产品前端实际上行的是 1000ms 一帧,模型每秒决策一次;按 80ms 发会把服务端的决策
次数抬到真实流量的 12.5 倍,压出来的容量偏悲观。所以帧长做成可配,压容量时用 1000。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

FRAME_MS = 80
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000


def samples_per_frame(frame_ms: int) -> int:
    return SAMPLE_RATE * frame_ms // 1000


@dataclass
class SpeechPattern:
    """说话活动的统计特征。

    默认值来自电话语音的实测,不是估的。Lee & Un 1986(IEEE Trans. Comm. 34(6))
    在 50 分钟电话语音上测得:平均说话段 1.11–1.30 秒,平均静音 2.22–2.41 秒,
    语音活动比 31.5–37.8%。下面的区间按这组数取,均值落在说话段约 1.2 秒、
    静音约 1.9 秒,算出来的活动比约 0.39。

    这一组比直觉值低得多,值得记一笔:拍脑袋设成说话段 3 秒、静音 1.75 秒的话,
    活动比是 0.63,上行负载差不多是真实情况的两倍,压出来的容量会明显偏悲观。
    说话段长度本身是重尾的(Brady 模型给对数正态、均值 4.14 秒、众数 0.25 秒),
    均匀分布只是粗略近似;要更准得换成对数正态采样。

    barge_in_probability 是「这一段静音里用户不等对方说完就插话」的概率。命中时
    静音直接截断到 near_zero_pause_s 以内,上行流变得密集。Brady 模型用「响应时间
    为负」定义打断,即第二个说话人在第一个还在说的时候开口,那是更精细的做法。
    """

    talk_min_s: float = 0.6
    talk_max_s: float = 1.8
    pause_min_s: float = 1.2
    pause_max_s: float = 2.6
    barge_in_probability: float = 0.0
    near_zero_pause_s: float = 0.05
    jitter_ms: float = 10.0
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample_talk_s(self) -> float:
        return self._rng.uniform(self.talk_min_s, self.talk_max_s)

    def sample_pause_s(self) -> float:
        if self.barge_in_probability > 0 and self._rng.random() < self.barge_in_probability:
            return self._rng.uniform(0.0, self.near_zero_pause_s)
        return self._rng.uniform(self.pause_min_s, self.pause_max_s)

    def sample_jitter_s(self) -> float:
        if self.jitter_ms <= 0:
            return 0.0
        return self._rng.uniform(-self.jitter_ms, self.jitter_ms) / 1000.0


class SpeechActivity:
    """按固定时间片推进的说话/静音状态机。

    调用方每隔一帧查询一次,得到本片是否要上行音频。状态推进用累积的时长
    而不是 sleep 累加,避免长会话里时钟漂移让所有客户端慢慢错开成同一个相位。
    """

    def __init__(self, pattern: SpeechPattern, start_offset_s: float = 0.0) -> None:
        self.pattern = pattern
        self._next_switch_at = start_offset_s + pattern.sample_talk_s()
        self._talking = True
        self._now = 0.0
        self.talk_switches = 0
        self.silence_switches = 0

    def advance(self, dt_s: float) -> bool:
        """推进 dt_s 秒,返回这段时间片内是否需要上行音频。"""
        self._now += dt_s
        while self._now >= self._next_switch_at:
            self._talking = not self._talking
            if self._talking:
                self.talk_switches += 1
                self._next_switch_at += self.pattern.sample_talk_s()
            else:
                self.silence_switches += 1
                self._next_switch_at += self.pattern.sample_pause_s()
        return self._talking

    @property
    def talking(self) -> bool:
        return self._talking


def load_pcm16k_mono(path: str) -> Tuple[bytes, int]:
    """读一个 16 kHz 单声道 PCM 的 WAV,返回 (raw float32 little-endian 字节, 帧数)。

    服务端要的是 base64 编码的 float32 PCM,不是 WAV 容器,所以这里用 wave 模块
    解析后重新打包。刻意不依赖 soundfile,让工具在没装音频栈的机器上也能跑。
    """
    import wave

    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"{path} 不是单声道,实时协议只接受 16 kHz 单声道输入")
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path} 的采样位宽是 {wf.getsampwidth() * 8} 位,需要 16 位 PCM")
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path} 的采样率是 {wf.getframerate()} Hz,需要 {SAMPLE_RATE} Hz")
        raw = wf.readframes(wf.getnframes())

    import array

    samples = array.array("h")
    samples.frombytes(raw)
    f32 = array.array("f", (s / 32768.0 for s in samples))
    return f32.tobytes(), wf.getnframes()


def slice_frames(pcm_bytes: bytes, frame_samples: int = SAMPLES_PER_FRAME) -> List[bytes]:
    """把整段 PCM 切成定长帧,不够一帧的尾部丢弃。"""
    frame_bytes = frame_samples * 4
    return [pcm_bytes[i : i + frame_bytes] for i in range(0, len(pcm_bytes) - frame_bytes + 1, frame_bytes)]
