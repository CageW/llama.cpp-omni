"""上行视频帧。

视频双工模式下客户端在 `input.append` 里带一个 `video_frames` 数组,元素是 base64 编码的
JPEG。服务端每个输入包只取第一帧,所以这里一次也只发一帧。

压测用什么画面不重要:视觉编码的开销由分辨率和切片数决定,与画面内容无关,所以同一批素材
循环发就够了。**分辨率重要**:同一套服务在 960×540 和 1920×1080 下的编码开销完全不是一
回事,而 JPEG 的尺寸写在文件头里、客户端不解析的话就不知道自己在发什么。所以这里解析 SOF
标记把宽高读出来记进结果,让报告能回答"这次压的是什么形状"。

不自己合成 JPEG:编码器要么依赖 Pillow(发压节点上没有,只有 numpy 和 websockets),要么
自己写一个基线编码器,而一个细节写错就会产出服务端解不开的文件——那时压测会静默退化成纯
音频,正是这套工具最该防住的那类错误。素材由调用方给,与上行音频 `--user-audio` 一致。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# JPEG 的起止标记。文件不是以 FFD8 开头就不是 JPEG,与其让服务端去解不开,不如这里就失败。
JPEG_SOI = b"\xff\xd8"
# 带尺寸的帧头标记(SOF0 基线、SOF1 扩展顺序、SOF2 渐进,以及它们的算术编码变体)。
# 跳过 SOF4(DHT)、SOF8(JPG)、SOF12(DAC)这几个不是帧头的。
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """从 JPEG 字节流里读出 (宽, 高);读不出来返回 None。

    只走标记链,不解码图像数据,所以不需要任何图像库。
    """
    if len(data) < 4 or data[:2] != JPEG_SOI:
        return None
    i = 2
    n = len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xFF, 0x01) or 0xD0 <= marker <= 0xD9:
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        if seg_len < 2:
            return None
        if marker in _SOF_MARKERS:
            if i + 9 > n:
                return None
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return width, height
        i += 2 + seg_len
    return None


def load_jpeg_frames(path: str) -> List[bytes]:
    """读入素材:一个 JPEG 文件,或者一个装着 JPEG 的目录(按文件名排序)。"""
    p = Path(path)
    if p.is_dir():
        files = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")
        )
        if not files:
            raise ValueError(f"目录 {path} 里没有 .jpg/.jpeg 文件")
    elif p.is_file():
        files = [p]
    else:
        raise ValueError(f"视频素材 {path} 不存在")

    frames: List[bytes] = []
    dims = None
    for f in files:
        data = f.read_bytes()
        if data[:2] != JPEG_SOI:
            raise ValueError(f"{f} 不是 JPEG(开头不是 FFD8);服务端只接受 JPEG 帧")
        d = jpeg_dimensions(data)
        if dims is None:
            dims = d
        elif d != dims:
            # 分辨率决定视觉编码开销。一组素材里混着不同尺寸,压出来的是几种负载的平均值,
            # 而报告只会记下第一张的尺寸,事后看不出来。
            raise ValueError(f"{f} 的分辨率 {d} 与前面的 {dims} 不同,一次压测的视频帧必须同尺寸")
        frames.append(data)
    return frames


@dataclass
class VideoSource:
    """按帧率决定每个上行包带不带视频帧,带就从素材里循环取一张。

    `fps` 与上行音频的帧长共同决定节奏:音频每 `frame_ms` 毫秒一个包,视频每秒 `fps` 帧,
    所以平均每 `1000 / (fps * frame_ms)` 个音频包带一帧。用累加器判断而不是取模,这样
    fps 与包率不成整数倍时长期平均仍然对得上(例如 80ms 的包配 1 fps 是每 12.5 个包一帧)。
    """

    frames: List[bytes]
    fps: float
    frame_ms: int
    max_slice_nums: int = 1

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("视频素材为空")
        if self.fps <= 0:
            raise ValueError("--video-fps 必须大于 0")
        self._credit = 0.0
        self._idx = 0
        self._b64 = [base64.b64encode(f).decode("ascii") for f in self.frames]

    @property
    def per_append(self) -> float:
        return self.fps * self.frame_ms / 1000.0

    @property
    def dimensions(self) -> Optional[Tuple[int, int]]:
        """首帧的宽高。多张素材尺寸不同时只报第一张,加载时已要求它们同尺寸。"""
        return jpeg_dimensions(self.frames[0])

    def next_frames(self) -> List[str]:
        """这个上行包该带的视频帧(base64),不该带就返回空列表。"""
        self._credit += self.per_append
        out: List[str] = []
        # 容差:fps 与包率不成整数倍时,累加的浮点尾差会让本该发的那一帧晚一个包,
        # 长期平均自洽,但一段固定长度的压测里发出的帧数会比声称的少一帧。
        while self._credit >= 1.0 - 1e-9:
            self._credit -= 1.0
            out.append(self._b64[self._idx % len(self._b64)])
            self._idx += 1
        # 服务端每个输入包只取第一帧,多发没有意义,反而把上行带宽和 base64 开销放大
        return out[:1]
