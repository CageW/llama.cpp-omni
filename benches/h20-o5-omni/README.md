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

## 2026-09-04 下午补充：部署形态搜索结果（Q8_0 权重、F16 KV、每 session 每秒 1 帧音频 + 1 帧图）

判定标准：每帧 LLM 决策延迟 P95 < 1 s。节点当天 load average 在 2 到 10 之间波动，executor 每 batch 耗时随主机负载变化 30% 以上，因此只在同批次内比较。所有布局都要设 `Omni_BACKEND_DEVICE=CUDA1`，否则 root 编码器会落到 LLM 所在卡。

| 布局（每个 LLM 副本一个进程） | 卡数 | session 总数 | 每副本 P50 / P95 (ms) | 超时帧 | 结论 |
|---|---|---|---|---|---|
| 1 LLM + 1 编码器 + 1 TTS/T2W | 3 | 16 | 771 / 1430 | 47/576 | 边缘（负载 4.7） |
| 1 LLM + 4 编码器副本 + 1 TTS/T2W，vision 合批 4 | 6 | 24 | 845 / 2060 | 122/864 | 不达标 |
| 同上 | 6 | 28 | 1112 / 15031 | 544/1008 | 不达标 |
| 2 LLM 副本，各自独占编码器卡与 TTS 卡 | 6 | 2×16 | 680 / 843，692 / 835 | 0 | 达标 |
| 2 LLM 副本，共用 1 张编码器卡 + 1 张 TTS 卡 | 4 | 2×16 | 685 / 829，683 / 817 | 3/1152 | 达标 |
| 同上 | 4 | 2×20 | 822 / 1271，804 / 1216 | 155/1440 | 边缘 |
| 3 LLM 副本，共用 1 张编码器卡 + 1 张 TTS 卡 | 5 | 3×16 | 731 / 869，803 / 920，814 / 989 | 31/1728 | 达标（负载 10） |

结论：单个 LLM 进程无论给多少编码器资源，实时上限都在 16 到 20 session；LLM 卡利用率只有 20%，GPU 时间（prefill batch 93 到 119 ms、decode batch 10 ms）只占每帧 LLM 墙钟（prefill 300 到 500 ms）的四分之一，其余是进程内 CPU 侧开销（每 token 对 248144 词表在主机侧采样、logits 拷贝、fused embedding 拼装、线程争用）。**多开 LLM 进程副本是当前最有效的扩容方式**，编码器和 TTS/Token2Wav 各一张卡可同时服务至少 48 session（利用率 4% 到 18%）。

按此外推 8 卡 H20：6 张 LLM 卡（每卡一个副本、16 session）+ 1 张编码器卡 + 1 张 TTS/Token2Wav 卡，约 96 个 omni session；若把每副本的 CPU 侧开销压下去让单副本到 24，则约 144。

已验证无效的方向：executor 固定形状 padding + MoE decode 走 CUDA graph（MoE decode 耗时随 batch 内 token 数增长，1 token 4.7 ms、16 token 16.5 ms，padding 反而变慢）；decode 优先调度（prefill 排队 1 s 以上）。

## 2026-09-04 晚：单进程 24 session 达标与 8 卡布局实测（新节点 10.20.20.10 / .6，8×H20，无其他租户）

上一节「单进程上限 16 到 20、瓶颈是主机侧开销」的结论已被推翻。用 nsys 按第一个 decode kernel 切分加载期与稳态后，稳态 1 秒窗口里 GPU kernel 忙 71%，一次 forward 内部几乎全是 kernel 执行；之前 nvidia-smi 上的 20% 是占空比（每个 session 每秒只推一帧），不是 kernel 效率。CUDA graph、减少 memcpy、采样优化三条路都没有收益，已关闭。

真正的浪费在合批宽度。24 个 session 每帧各一次 79 token 的 prefill，合成一批只要 8 ms 一个，单独跑要 77 ms 一个；此前平均每批只合到 3.1 个，prefill 占了运行时间的 31/36 秒。两个原因叠加：一是 executor 的合批窗口只等 3 ms；二是编码器与 LLM 同卡时抢算力，各 session 的编码完成时刻在一个 tick 内散开 275 到 483 ms，prefill 天然到不齐。「多等一会儿凑批」的所有参数都更差，因为 session 要等本帧 decode 完成才提交下一帧 prefill，而 decode 排在同一队列里被凑批等待挡住。

修法两条，缺一不可：`OMNI_EXEC_POLICY=ripe`（decode 永不扣住；prefill 扣到凑齐 `OMNI_EXEC_PREFILL_TARGET` 个或最早的等了 `OMNI_EXEC_PREFILL_DEADLINE_US`=200 ms 才整批放行）；编码器搬离 LLM 卡（`Omni_BACKEND_DEVICE=CUDA1` 加 `--enc-devices` 副本，`OMNI_VISION_BATCH=1 OMNI_VISION_BATCH_MAX=4`）。搬离后编码完成散开只剩 88 ms，prefill 宽度到 24.0。

单进程（1 张 LLM 卡）的 session 上限，均含 Q8_0 权重、F16 KV：

| session | TTS | 编码器卡 | P50 / P95 (ms) | 超时帧 | 结论 |
|---|---|---|---|---|---|
| 20 | 有 | 3 | 732 / 898 | 2/720 | 达标，余量大 |
| 24 | 有 | 3 | 856 / 1119 | 98/864 | 达标 |
| 24 | 无 | 1 | 1048 / 1312 | 498/864 | 不达标：1 张编码器卡撑不住 24 |
| 26 | 无 | 3 | 953 / 1349 | 299/936 | 边缘 |
| 28 | 无 | 3 | 1104 / 2358 | 672/1008 | 不达标 |
| 32 | 无 | 3 | 1879 / 5655 | 905/1152 | 不达标 |

单进程上限是 24。再往上不是吞吐不够（executor 只忙 44%），是延迟链：24 宽的 prefill 批本身 230 ms，加 200 ms 凑批 deadline，加 4 次 decode 往返，单 session 的串行链已顶到 1 秒。同一张卡开 2 个 executor 更差（prefill 被拆进两个队列，宽度只有 8，P95 6.3 s）。

多进程布局。**这一节之前的版本作废**：多个进程各自加载 35 GB 模型，稳态阶段几乎不重叠（三进程的第三个常在前两个跑完后才开始），所以「3×24 边缘达标」是把 2 个并发加 1 个独跑当成了 3 个并发。现在 `llama-omni-perf-multisession` 加了 `--start-file`：每个进程加载并 session_begin 完成后打印 `[ready]` 并等待同一个文件出现，外部脚本等所有进程就绪后 touch 它，三个进程在同一秒开始推帧；统计只取日志里 `[go]` 之后的段。两台节点对调同一布局，结果一致。

| 布局（含 TTS） | 卡数 | session | 各进程 P95 (ms) | 超时帧 | 结论 |
|---|---|---|---|---|---|
| 3 LLM + 3 编码器 + 2 语音，3×20 | 8 | 60 | 1095 / 1113 / 1027（另一台 1132 / 1080 / 1045） | 13% | 边缘 |
| 3 LLM + 3 编码器 + 2 语音，3×24 | 8 | 72 | 1312 / 1369 / 1411 | 78% | 不达标 |
| 2 LLM + 2 编码器 + 2 语音，2×24 | 6 | 48 | 1826 / 1352（另一台 1969 / 1760） | 67% | 不达标 |

「语音卡」指 TTS 的小 LLM 与 Token2Wav 同卡；两张语音卡时 Token2Wav 按 session 对半拆（`--t2w-devices gpu:X,gpu:Y`）。

上面三行都是三个进程的帧 tick **同相位**跑出来的：60 个 session 的图像在同一瞬间打到编码器卡。给 `llama-omni-perf-multisession` 加了 `--start-delay-ms`，让三个进程的 tick 错开三分之一周期（0 / 333 / 666 ms）之后，同样的布局结果完全不同：

| 布局（含 TTS，进程间错峰） | 卡数 | session | 各进程 P95 (ms) | 超时帧 | 结论 |
|---|---|---|---|---|---|
| 3 LLM + 3 编码器 + 2 语音，3×20 | 8 | 60 | 893 / 883 / 893（重复一次：897 / 883 / 911） | 0.9% | **达标** |
| 3 LLM + 3 编码器 + 2 语音，3×24 | 8 | 72 | 1140 / 1052 / 1012 | 9% | 边缘 |
| 3 LLM + 2 编码器 + 3 语音，3×24 | 8 | 72 | 1121 / 1231 / 1210 | 28% | 不达标 |
| 2 LLM + 2 编码器 + 2 语音，2×24（错峰 500 ms） | 6 | 48 | 1758 / 1124 | 21% | 一个达标一个不 |

错峰对编码器 stage 的影响是决定性的：同一张编码器卡在同相位 20 个 session 时利用率 97%、一批 4 张图的 GPU 时间 165 ms；错峰后利用率 11% 到 69%、同一批只要 59 ms，各 session 的编码完成时刻在一个 tick 内的散开从 165 到 205 ms 缩到 59 ms，LLM 侧的 prefill 合批宽度回到满宽。这个 bench 里进程内的 session 严格同相位、只在进程间错开，对生产（session 到达相位天然随机）是偏乐观的近似；反过来说，生产里的随机相位也会让进程内的 prefill 合批变窄。

三个 stage 各自的每卡容量，从错峰实测收敛出来的：LLM 卡 24 个 session；编码器卡 20 个（同相位）到 30 个以上（错峰，72 时每卡 24 个只用了 27% 到 38%）；**语音卡约 30 个**，这是最硬的一条：72 个 session 摆 2 张语音卡时两张 99.7% / 92%，摆 3 张时 73% / 72% / 87% 反而更差（TTS-LLM 每进程独占一张、Token2Wav 三路拆分，多进程多 context 争一张卡的 time-slice 又回来了）。**这个架构在 8 卡 H20 上的结论：60 个 omni session 达标，72 边缘。** 比最早按 16/进程外推的 96 低，因为那次把编码器和语音 stage 当成了无限资源。

要再往上，杠杆已经不在 LLM，而在 Token2Wav：每卡约 30 个 session 就满，8 卡里已经要拿 2 到 3 张给它。
