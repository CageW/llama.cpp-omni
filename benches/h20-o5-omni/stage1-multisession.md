# 阶段 1 与阶段 3 初测：MiniCPM-o5 多 session 共享权重（2026-09-04，南京 H20）

工具：`llama-omni-perf-multisession`。一份权重，N 个 duplex session 并发，每 session 每秒推一帧
1 s 音频 + 一帧 960×540 图，共 36 帧。LLM Q8_0，KV q8_0，n_ctx 8192，TTS 上下文 2048。
判据：每帧 LLM 判定延迟（push 到 listen/speak）P95 低于 1 s 进帧间隔。

## 显存

| 项 | 大小 |
|---|---|
| 全部模块权重 + 第 1 个 session | 40.6 GB |
| 每增加 1 个 session（F16 KV） | 2.82 GB |
| 每增加 1 个 session（q8_0 KV，TTS ctx 2048） | 2.33 GB |
| 其中计算缓冲（LLM 497 + TTS 486 + APM 88 MB） | 1.07 GB |
| 其中 LLM KV（8K，q8_0） | 85 MB |
| 其中 gated-delta-net 递归状态 | 63 MB |
| 其中 TTS KV（2048，q8_0） | 64 MB |
| 其中 APM 流式 KV（1500 帧） | 141 MB |

纯显存上限：(96 − 40.6) / 2.33 ≈ 23 个 session 每卡。实际上限由下面的算力排队决定。

## 单卡并发延迟（每 session 独立上下文、不做跨 session batching）

| 并发 | P50 | P95 | 超 1 s 帧数 | 说明 |
|---|---|---|---|---|
| 1 | 119 ms | 158 ms | 0 / 36 | |
| 2 | 175 ms | 219 ms | 0 / 72 | |
| 4 | 288 ms | 358 ms | 0 / 144 | |
| 8 | 506 ms | 671 ms | 0 / 288 | 通过 |
| 12 | 880 ms | 2.3 s | 187 / 432 | 不通过 |
| 16 | 6.0 s | 16 s | 550 / 576 | 不通过 |
| 16，关 TTS | 2.4 s | 7.1 s | 508 / 576 | 只剩 LLM + 编码器仍不通过 |

以上为每 session 各自持有 vision、APM、Token2Mel 的 CUDA backend 实例（各自 stream 与内存池）之后的数字。
共用一个 backend 实例并加互斥锁时，8 并发 P95 725 ms，12 并发 P95 2.2 s，差别不大。

## 分项耗时随并发的变化（单卡）

| 并发 | LLM fused prefill 平均 | 视觉编码平均 | Token2Wav 每 chunk 平均 |
|---|---|---|---|
| 1 | 约 80 ms | 61 ms | 54 ms |
| 8 | 196 ms | 149 ms | 886 ms |
| 12 | 237 ms | 244 ms | 2.8 s |
| 16 | 219 ms | 396 ms | 4.0 s |

## Token2Wav 放到别的卡（阶段 3 的第一种放置）

| 布局 | 并发 | P50 | P95 | Token2Wav 每 chunk |
|---|---|---|---|---|
| 全部同卡 | 12 | 880 ms | 2.3 s | 2.8 s |
| Token2Wav 分到 2 卡 | 12 | 795 ms | 1.9 s | 777 ms |
| Token2Wav 分到 3 卡 | 16 | 5.0 s | 18 s | 1.1 s |

Token2Wav 的排队被分掉了，但 LLM 判定延迟几乎没变，因为 LLM prefill 和视觉编码本身已经把主卡填满。

## 结论

1. 阶段 1 的目标达成：一进程多 session、权重只加载一份，每 session 增量 2.3 到 2.8 GB。
2. 不做跨 session batching 时，单卡 omni 模式的实时上限是 8 到 10 个 session。这与用同形状 LLM 在 batched-bench 上测出的「batch=1 每帧 prefill 60 到 90 ms」一致：每秒 16 帧 prefill 的 GPU 时间已超过 1 s。nvidia-smi 报的 33% 到 38% 利用率是粗粒度采样，真实情况是 kernel 发射受限、SM 占用低。
3. 把 Token2Wav 挪到别的卡有效但不是当前的墙，LLM prefill 和视觉编码才是。两者都有现成的 batch 路径可走：LLM 把所有 session 当前秒的 prefill 合成一个多 sequence 的 batch（同形状实测 batch 8 时 3760 tok/s，够 36 个 omni session）；视觉编码用已有的 `vision_image_batch_encode` 把同尺寸帧合成一张图。
4. 因此接下来的顺序改为：先做阶段 2（LLM 跨 session prefill batching 加视觉 batch），再回到阶段 3 的多卡放置。没有阶段 2，多卡放置只能把 8 卡乘 8 到 10 个 session 拼成 64 到 80 个，远低于显存允许的 180 个。
