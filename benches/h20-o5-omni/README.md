# MiniCPM-o5 在 8 卡 H20 上的多 session 容量：实测结论（2026-09-04）

分支 `exp/o5-multisession-capacity`。测试机：南京节点，8×H20 96 GB，NV18 全互联。
每个 session 每秒推一帧：omni 模式为 1 s 音频 + 一帧 960×540 图（LLM 每秒约 80 token），
纯语音模式只有 1 s 音频（每秒约 17 token）。判据：每帧 LLM 判定延迟（推帧到 listen/speak）
P95 低于 1 s 进帧间隔。LLM Q8_0，KV q8_0，n_ctx 8192，TTS 上下文 2048，TTS/LLM ubatch 64/128。

## 结论

| 模式 | 单卡实时上限 | 限制因素 | 8 卡估计 |
|---|---|---|---|
| omni（音频 + 视频） | 8 个 session（10 个时有 6/360 帧超时） | MoE prefill 算力：单卡 3200 到 3700 tok/s，每 session 每秒 80 token | 64 个 |
| 纯语音 | 16 个 session（24 个时 P95 3.1 s） | Token2Wav 排队，其次 LLM prefill | 128 个 |
| 纯语音，Token2Wav 分到别的卡 | 16 个通过，24 个 P95 2.8 s | LLM prefill 加上视觉之外的编码开销 | 约 130 个 |

8 卡估计已用 6 张空闲卡同时各跑 8 个 omni session 验证：六张卡的 P95 全部在 650 到 686 ms，互不干扰。

这些数字与阶段 0 之前用显存推出的「约 1000 个 session」相差一个数量级，原因是那次假设了算力充足。
o5 的 LLM 是 35B-A3B 的 MoE，每 token 激活 3B 参数，在 H20 上 prefill 吞吐 3.2 到 3.7 千 token 每秒，
且从 batch 4 起就平了：合批不能再提高吞吐，只能把各 session 的时间加在一起。

## 各阶段做了什么

阶段 0：o5 跑通。转换脚本 `tools/omni/convert/convert_o5.py`；胶水层去掉 4096 维和 o4.5 词表的硬编码；
M-RoPE 位置按 4 段广播；o5 关闭滑窗（hybrid memory 不能删中间区间）。语音与 omni 模式输出经 ASR 校验可懂。

阶段 1：一进程多 session、权重共享。`omni_session_clone` 复用五个模块的权重，每 session 只持有上下文、
流式 KV、Token2Wav 流式状态和线程。每 session 增量从 2.82 GB 压到 1.69 GB（KV q8_0、TTS ctx 2048、
ubatch 64/128）。共享同一个 ggml CUDA backend 实例会打爆它的内存池，改为每 session 自己的 backend
实例（vision、APM、Token2Mel），vocoder 因内部常量张量在 eval 时重分配，仍共用并加锁。

阶段 2：LLM 跨 session batching。写了共享执行器（一个 llama_context、N 个 sequence、一个调度线程），
并改核心库让显式输出标记下的 embedding 不再强制全 token 输出、hybrid memory 不再按 sequence 拆 ubatch。
结果：8 个 omni session 的 P95 从 671 ms 变成 1.48 s，更差。执行器 trace 显示 prefill 批的时间与 batch 内
token 数近似成比例（77 token 33 ms，283 token 172 ms），与 batched-bench 的曲线一致。batch 上的 MoE
在这张卡上是算力墙，不是发射墙。CUDA graph 开关、KV unified 开关、decode 优先调度都不改变结论。

阶段 3：stage 拆分放置。Token2Wav 权重可在别的卡加载并按 session 轮转（`--t2w-devices`）。
omni 模式下把 Token2Wav 分到 2 张卡，12 个 session 的 P95 从 2.3 s 变 1.9 s，Token2Wav 每 chunk 从
2.8 s 降到 0.78 s，但 LLM 判定延迟基本不动，因为墙在 LLM prefill。语音模式 24 个 session 分 3 卡，
Token2Wav 每 chunk 降到 0.5 s，P95 仍 2.8 s，墙同样在 LLM。

## 关键数字

单卡显存（Q8_0 LLM）：

| 项 | 大小 |
|---|---|
| 全部模块权重 + 1 个 session | 39.8 GB |
| 每增一个 session（最终配置） | 1.69 GB |
| 其中 LLM 计算缓冲（ubatch 128） | 124 MB |
| 其中 TTS 计算缓冲（ubatch 64） | 61 MB |
| 其中 APM 计算缓冲 + 流式 KV | 88 + 141 MB |
| 其中 LLM KV（8K，q8_0） + 递归状态 | 85 + 63 MB |
| 其中 TTS KV（2048，q8_0） | 64 MB |
| 其余（Token2Wav 流式状态、各 backend 内存池、采样器） | 约 1.1 GB |

显存上限：(96 − 39.8) / 1.69 ≈ 33 个 session，与算力上限的差距就是「不做 batching 的代价」。

单卡 omni 并发延迟（每 session 独立上下文）：

| 并发 | P50 | P95 | 超 1 s |
|---|---|---|---|
| 1 | 119 ms | 158 ms | 0 |
| 4 | 288 ms | 358 ms | 0 |
| 8 | 508 ms | 679 ms | 0 / 288 |
| 10 | 649 ms | 897 ms | 6 / 360 |
| 12 | 880 ms | 2.3 s | 187 / 432 |
| 16 | 6.0 s | 16 s | 550 / 576 |

单卡纯语音：

| 并发 | P50 | P95 | 超 1 s |
|---|---|---|---|
| 8 | 159 ms | 386 ms | 0 / 288 |
| 16 | 328 ms | 567 ms | 0 / 576 |
| 24 | 558 ms | 3.1 s | 234 / 864 |

MoE prefill 墙与权重格式的关系（batched-bench，77 token prefill，token/s）：

| batch | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| 1 | 816 | 1415 | 1338 |
| 4 | 1896 | 3159 | 2942 |
| 8 | 2427 | 3168 | 2925 |
| 16 | 2353 | 3571 | 3281 |

Q4_K_M 不比 Q8_0 快，F16 更慢。墙不是权重读取带宽，是 MoE 专家矩阵乘在小 batch 下的 kernel 效率。

## 要提高上限该做什么

按收益排序：

1. MoE prefill kernel。8 个 session 时 GPU 利用率只有 30% 到 38%，SM 大量空闲。ggml 的 MUL_MAT_ID 在
   batch 大于 8 时走的路径对 H20 的 Q8_0 权重只有 3.5 千 token 每秒，同样形状的模型在 PyTorch 用
   batched_mm 是另一条路径。这是唯一能把 omni 上限从 8 抬到 30 的杠杆，工作在 ggml-cuda 里。
2. 视觉编码合批。8 个 session 时视觉编码从 61 ms 涨到 150 ms，`vision_image_batch_encode` 已支持同尺寸
   帧合批，接到调度层即可，对 omni 模式约省 100 ms。
3. Token2Wav 的 vocoder 常量张量改为一次分配，让 vocoder 也能每 session 一个 backend 实例。语音模式的
   16 到 24 之间会因此松动。
4. 多卡放置只在 1 做完后才有意义。做完 1 之后每卡 omni 上限约 30，8 卡约 240，此时 Token2Wav 和视觉
   编码会成为下一道墙，`--t2w-devices` 那套放置就能发挥作用。

## 复现

```
# 构建（节点上）
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DLLAMA_CURL=OFF
cmake --build build-cuda -j --target llama-omni-perf-multisession llama-omni-cli llama-omni-server
# 单卡 N 个 omni session
OMNI_TTS_CTX=2048 OMNI_TTS_UBATCH=64 OMNI_LLM_UBATCH=128 CUDA_VISIBLE_DEVICES=0 \
  build-cuda/bin/llama-omni-perf-multisession -m <dir>/MiniCPM-o5-LLM-Q8_0.gguf \
  --test tools/omni/assets/test_case/duplex_omni_test_case/duplex_omni_test_case_ 36 \
  -n 8 --interval 1000 -c 8192 --ctk q8_0 --ub 128
# Token2Wav 放到别的卡：CUDA_VISIBLE_DEVICES=0,1 ... --t2w-devices gpu:0,gpu:1
# 共享执行器（合批实验）：--exec，OMNI_EXEC_POLICY=decode / OMNI_EXEC_TRACE=1
```
GGUF 在南京节点 `/data/wangkaiqi/o5gguf/`，原始日志 `/data/wangkaiqi/logs/ms-*.log`，结果 CSV `/data/wangkaiqi/o5out/ms/results_*.csv`。
