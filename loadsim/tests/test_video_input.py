"""音视频上行:帧真的发出去了,而且服务端真的处理了。

这套测试的重点不是「能不能发」,是「发了但没生效时能不能被发现」。服务端把 video_frames
丢掉的时候,客户端侧一切正常:延迟、吞吐、断流都正常,只是测的是纯音频负载。这类静默退化
在本项目里已经出过几次,所以判定里有一条硬门禁,这里逐条验它。
"""

import asyncio
import base64
from pathlib import Path

import pytest

from loadsim.fake_server import serve_fake
from loadsim.runner import RunConfig, run_load
from loadsim.video import VideoSource, jpeg_dimensions, load_jpeg_frames

# 32x24 的基线 JPEG,679 字节。内联成 base64 而不是放二进制素材文件,是为了让这套测试
# 不依赖任何图像库——发压节点上只有 numpy 和 websockets,没有 Pillow。
TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABQODxIPDRQSEBIXFRQYHjIhHhwcHj0sLiQySUBMS0dARkVQWnNiUFVtVkVGZIhlbXd7gYKBTmCNl4x9lnN+gXz/2wBDARUXFx4aHjshITt8U0ZTfHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHz/wAARCAAYACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDm/sXtS/Yvauh+xe1L9i9qAOe+xe1L9i9q6H7F7Uv2L2oA3/sXtS/YvaiigBfsXtS/YvaiigD/2Q=="
)
TINY_JPEG = base64.b64decode(TINY_JPEG_B64)


@pytest.fixture
def jpeg_file(tmp_path: Path) -> str:
    f = tmp_path / "frame.jpg"
    f.write_bytes(TINY_JPEG)
    return str(f)


def test_jpeg_dimensions_read_from_the_file_header():
    assert jpeg_dimensions(TINY_JPEG) == (32, 24)
    assert jpeg_dimensions(b"not a jpeg") is None


def test_non_jpeg_material_is_rejected_at_load(tmp_path: Path):
    bad = tmp_path / "frame.jpg"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    with pytest.raises(ValueError, match="JPEG"):
        load_jpeg_frames(str(bad))


def test_mixed_resolutions_are_rejected(tmp_path: Path, jpeg_file: str):
    # 第二张尺寸不同:把 SOF 里的高度改掉就够了,不需要重新编码
    other = bytearray(TINY_JPEG)
    i = other.find(b"\xff\xc0")
    other[i + 5:i + 7] = (48).to_bytes(2, "big")
    (tmp_path / "b_frame.jpg").write_bytes(bytes(other))
    with pytest.raises(ValueError, match="分辨率"):
        load_jpeg_frames(str(tmp_path))


@pytest.mark.parametrize(
    "fps,frame_ms,appends,expected",
    [(1.0, 1000, 5, 5), (0.5, 1000, 10, 5), (1.0, 80, 25, 2)],
)
def test_frame_pacing_matches_the_declared_fps(fps, frame_ms, appends, expected):
    vs = VideoSource(frames=[TINY_JPEG], fps=fps, frame_ms=frame_ms)
    assert sum(len(vs.next_frames()) for _ in range(appends)) == expected


def _run(port: int, jpeg: str, input_mode: str = "audio-video") -> dict:
    cfg = RunConfig(
        url=f"ws://127.0.0.1:{port}/v1/realtime?mode=video",
        arrival_mode="ramp", target_concurrency=2, duration_s=6.0, ramp_s=0.5,
        lifetime_mean_s=4.0, lifetime_dist="fixed", lifetime_min_s=1.0, lifetime_max_s=8.0,
        seed=3, frame_ms=1000, input_mode=input_mode, video_source_path=jpeg, video_fps=1.0,
        max_slice_nums=2,
    )
    return cfg


def test_video_frames_are_sent_and_confirmed_by_the_server(jpeg_file: str):
    async def go():
        _s, port, stop = await serve_fake(echo_input=True, report_vision=True)
        try:
            return await run_load(_run(port, jpeg_file), frames=[b"\x00" * 32000])
        finally:
            await stop()

    rep = asyncio.run(go())
    s = rep.summary
    assert s["input_mode"] == "audio-video"
    assert s["video_frames_sent_total"] > 0
    # max_slice_nums=2,假服务端按切片数回报,所以切片数应当是帧数的两倍
    assert s["vision_slices_total"] == 2 * s["video_frames_sent_total"]
    assert s["vision_tokens_total"] == 64 * s["vision_slices_total"]
    assert "纯音频负载" not in " ".join(s["verdict_reasons"])


def test_server_dropping_video_is_caught(jpeg_file: str):
    """服务端把 video_frames 丢掉时必须判 fail,而不是给出一份看着正常的报告。"""
    async def go():
        _s, port, stop = await serve_fake(echo_input=True, report_vision=False)
        try:
            return await run_load(_run(port, jpeg_file), frames=[b"\x00" * 32000])
        finally:
            await stop()

    rep = asyncio.run(go())
    s = rep.summary
    assert s["video_frames_sent_total"] > 0
    assert s["vision_slices_total"] == 0
    assert s["verdict"] == "fail"
    assert any("纯音频负载" in r for r in s["verdict_reasons"])


def test_audio_only_mode_sends_no_video(jpeg_file: str):
    async def go():
        _s, port, stop = await serve_fake(echo_input=True, report_vision=True)
        try:
            cfg = _run(port, jpeg_file, input_mode="audio")
            cfg.video_source_path = None
            return await run_load(cfg, frames=[b"\x00" * 32000])
        finally:
            await stop()

    rep = asyncio.run(go())
    s = rep.summary
    assert s["input_mode"] == "audio"
    assert s["video_frames_sent_total"] == 0
    assert not any("视频" in r for r in s["verdict_reasons"])


def test_audio_video_without_material_fails_fast(jpeg_file: str):
    async def go():
        _s, port, stop = await serve_fake()
        try:
            cfg = _run(port, jpeg_file)
            cfg.video_source_path = None
            return await run_load(cfg, frames=[b"\x00" * 32000])
        finally:
            await stop()

    with pytest.raises(ValueError, match="video_source_path"):
        asyncio.run(go())
