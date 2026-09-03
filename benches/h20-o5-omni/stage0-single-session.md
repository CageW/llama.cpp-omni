# 阶段 0：MiniCPM-o5 在 llama.cpp-omni 单 session 跑通（2026-09-04，南京 H20 单卡）

模型：LLM Q8_0（36.9 GB），vision/audio/tts F16，Token2Wav 复用 o4.5 GGUF（权重逐字节相同）。
构建：CUDA 13.0，sm_90，flash attention 开。GPU：H20 96 GB，单卡。

## 语音模式（`--test audio_test_case_ 2`）

| 项 | 值 |
|---|---|
| LLM prefill（system prompt 63 token + 10 音频 token） | 约 60 ms |
| LLM decode 每 token | 约 6 ms |
| TTS 每 token | 约 4 ms |
| Token2Wav 每 1 s 音频 chunk | 54 ms（RTF 0.05） |
| 首响（首个 wav 落盘） | 410 ms |
| ASR（paraformer-zh）对 26 s 输出 | 全句可懂，与 LLM 文本一致 |

## Omni 全双工（`llama-omni-perf-duplex`，36 帧，每帧 1 s 音频 + 1 帧 960×540 图，进帧间隔 1000 ms）

| 项 | 值 |
|---|---|
| 每帧 LLM 判定延迟（push 到 listen/speak） | P50 118 ms，P95 148 ms，max 154 ms |
| 每帧 prefill token 数 | 约 81（unit 标记 + 66 视觉 + 10 音频 + 结构 token） |
| 视觉编码每帧 | 61 到 70 ms（单 overview chunk，66 token） |
| speak 首响 e2e | 411 ms |
| TTS RTF | 0.14 |
| 36 帧内 speak 轮次 | 3，全部有音频，ASR 可懂且与场景相符 |

结论：单 session 在 H20 上满足 duplex 实时性（三项判据全部通过）。进帧间隔设 0 时 LLM 判定 P95 3.7 s，这是 TTS 线程与 LLM 线程抢同一张卡加上 TTS 队列积压，不是模型问题。

## 与 PyTorch 参考的差异（已核实）

- ViT merger 的窗口注意力：C++ 用 4×4 block-diagonal 窗口 mask（flash-attn varlen 语义，也是上游 mtmd 的实现）；南京节点上的 PyTorch 参考因无 flash_attn 走 eager 路径，merger 的 attention_mask 为 None，实际是全局注意力。两者对同一像素输入的最终视觉嵌入 cos 0.9635；C++ 对「加窗口 mask 的 Python」cos 0.9999。ViT 前 7 层、pos-embed、merger 之后各段逐层 cos ≈ 1.0。
- TTS 条件（emb_text + l2norm(projector_semantic(hidden))）公式与 Python 一致。
- o5 每帧只用单 overview chunk（对应 Python streaming_prefill 默认 max_slice_nums=1）；带 slice 的 3-D canvas M-RoPE 位置尚未接入 C++ prefill，开 slice 会退化为文本位置，导致输出乱码。

## 未做

- teacher-forced 逐 token logits 对比（需要 Python 侧 dump 脚本，留待阶段 2 batching 回归时一起做）。
- 长对话到 n_ctx 的 reset 策略（o5 hybrid memory 不支持中间区间删除，滑窗已对 o5 关闭）。
