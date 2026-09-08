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

三个 stage 各自的每卡容量，从错峰实测收敛出来的：LLM 卡 24 个 session；编码器卡 20 个（同相位）到 30 个以上（错峰，72 时每卡 24 个只用了 27% 到 38%）；**语音卡约 30 个**，这是最硬的一条：72 个 session 摆 2 张语音卡时两张 99.7% / 92%。把第三张语音卡从编码器池里拿过来（2 编码器 + 3 语音）时语音卡降到 73% / 72% / 87%，但编码侧回到了瓶颈（编码完成散开 91 到 155 ms，视觉编码 147 到 173 ms，prefill 合批宽度 24 到 22），整体更差。72 个 session 需要编码器和语音各 3 张卡，加上 3 张 LLM 卡是 9 张，8 张卡凑不出来。**这个架构在 8 卡 H20 上的结论：60 个 omni session 达标，72 边缘。** 比最早按 16/进程外推的 96 低，因为那次把编码器和语音 stage 当成了无限资源。

要再往上，杠杆已经不在 LLM，而在 Token2Wav：每卡约 30 个 session 就满，8 卡里已经要拿 2 到 3 张给它。

试过一个不改代码的办法：把编码器和 Token2Wav 混放在同一批卡上（3 张 LLM 卡加 5 张混合卡），指望用编码器卡的空闲填语音卡的缺口。72 个 session 错峰跑，两台节点都不达标（P95 1175 到 1315 ms，超时约 30%），比分开摆更差。原因是 Token2Wav 的 kernel 把同卡编码器的尾延迟顶了上去，视觉编码最大值从 280 ms 涨到 600 到 966 ms，编码完成时刻的散开翻倍，LLM 侧的 prefill 合批宽度从 24 掉到 15 到 18。编码器这个 stage 对同卡邻居的突发很敏感，要独占卡。

## 2026-09-05 上午：TTS 跨 session 合批之后，8 卡到 80 个 session

上一节的 60 是在「TTS 小模型每个 session 一个独立上下文、逐 token 单独跑」的前提下测的。这个 20 层的 LLaMA-768 在 72 个 session 时单独占一张卡 61%，而它的每一步只是一个 token 的 forward，正是主干 LLM 合批之前的状态。把主干 LLM 那套共享执行器搬过来（一个上下文、所有 session 各占一个 sequence、同一时刻的步合成一个 forward），TTS 的 GPU 时间降到每进程每 36 秒 4 秒，可以和主干 LLM 放在同一张卡上。

这一步腾出的卡拿来多开一个 LLM 进程，每进程的 session 数从 24 降到 20，单进程的延迟链随之松下来。8 张卡的分法变成：4 张 LLM 卡（每张一个进程，主干 LLM 与 TTS 的执行器同卡），2 张编码器卡，2 张 Token2Wav 卡。四个进程的帧节拍错开四分之一秒。

| 布局（含 TTS，进程间错峰，barrier 同起） | 卡数 | session | 各进程 P95 (ms) | 超时帧 | 结论 |
|---|---|---|---|---|---|
| 4 LLM + 2 编码器 + 2 Token2Wav，4×18 | 8 | 72 | 955 / 849 / 891 / 869 | 0.9% | 达标 |
| 4 LLM + 2 编码器 + 2 Token2Wav，4×20 | 8 | 80 | 981 / 946 / 949 / 967 | 3.3% | 达标 |
| 3 LLM + 3 编码器 + 2 Token2Wav，3×24，TTS 合批放 LLM 卡 | 8 | 72 | 1146 / 1166 / 1092 | 14% | 边缘 |
| 3 LLM + 3 编码器 + 2 语音卡，3×24，TTS 合批放 Token2Wav 卡 | 8 | 72 | 6421 / 1202 / 5939 | 39% | 不达标 |

后两行说明两件事：TTS 的执行器不能和 Token2Wav 同卡（Token2Wav 把卡占到 88% 后，TTS 那些 1 毫秒的小 forward 被拉长五倍，整条语音链积压）；而放在 LLM 卡上只多占 14% 的 GPU 时间。另一条不改代码的路（把 TTS 的独立上下文直接放到 LLM 卡）不可行，24 个各自解码的 TTS 上下文会把主干 LLM 的 forward 挤碎，P95 到 2 到 4 秒。

80 在另一台节点重复一次：P95 1041 / 954 / 983 / 956，超时 4.2%，可复现。再加到 4×22 = 88 时 P95 1001 到 1094，超时 7.9%，属于边缘。

80 个 session 时的利用率：LLM 卡 66% 到 86%，Token2Wav 卡 87%，编码器卡 49% 到 60%；88 时 Token2Wav 卡到 90%。下一个卡住的是 Token2Wav：它现在每个 session 的每个音频 chunk 单独过一遍 flow matching 和声码器，没有跨 session 的合批。

## 2026-09-05 下午：Token2Wav 跨 session 串行的根因与修法；上限回到 LLM 卡的延迟链

上一节末尾把下一个瓶颈指向 Token2Wav「每个 chunk 单独过一遍 flow matching 和声码器」。先量了它到底慢在哪。单独一张 Token2Wav 卡放 24 个 session（`OMNI_T2W_PROFILE` 汇总），每个 chunk 的纯计算只有 60 ms：token2mel（conformer 编码器 + 5 步 flow matching）p50 34 ms，声码器 p50 16 ms。但 Token2Wav 线程看到的每 chunk 推理时延 p50 是 857 ms，排队 p90 1.6 s。差的那 800 ms 不是算力，是锁：`push_tokens_window` 在整个 chunk 调用期间同时持有 token2mel backend 和声码器 backend 的两把互斥锁，token2mel 的 backend 是每 session 独立的，声码器 backend 却是所有 session 共用父 session 的那一个，于是 24 个 session 的整段 chunk 都在这把锁上排成一列，一次突发 24 × 60 ms 约 1.4 s。8 个 session 时没这个问题（p50 108 ms），所以之前按单 session 或少 session 的 profile 看不出来。

以前试过给声码器每 session 独立 backend，会在 `ggml_backend_tensor_set` 上断言 tensor buffer not set，当时归因于 STFT 常量张量的 buffer 归属，因此保留了共享 backend。这次复现后看清了原因：声码器每次调用都在本次调用的 ggml 上下文里重建 STFT 窗、DFT 表和正弦发生器的常量张量，再把指针写回共享的 generator 结构体。有整段大锁时这是串行的，没问题；一旦各 session 有了自己的 backend、不再被同一把锁串行化，两个 session 并发建图就互相覆盖这些指针，其中一个再去上传常量时张量已经不属于自己的图。与 backend 本身无关。

修法三处：每个 session 的声码器 runner 持有 generator 结构体的一份私有副本（权重张量指针仍指向共享缓冲区，只有那些每次调用重建的常量指针变成私有）；克隆 session 的声码器独立 backend 改为默认开启（`OMNI_VOC_OWN_BACKEND=0` 回退共享）；`push_tokens_window` 改成 token2mel 只锁自己的 backend、算完即放，声码器另起一段锁。改完在同一张卡上重跑 24 个 session：每 chunk 推理 p50 从 857 ms 降到 117 ms，排队 p90 从 1.6 s 归零。并发之后各 stage 自己的耗时变长（token2mel 34 到 67 ms、声码器 16 到 30 ms），那是 GPU 分时，不是回退。每 session 多约 130 MiB 显存（独立的内存池和分配器）。

然后回到 8 卡布局。Token2Wav 卡的利用率如预期下来了，但 session 上限没有动：

| 布局（含 TTS，进程间错峰，barrier 同起） | session | 各进程 P95 (ms) | 超时帧 | Token2Wav 卡利用率 | 结论 |
|---|---|---|---|---|---|
| 4 LLM + 2 编码器 + 2 Token2Wav，4×20 | 80 | 962 / 969 / 995 / 979 | 3.5% | 70% / 77%（修前 87% / 90%） | 达标，与修前持平 |
| 同上，4×22 | 88 | 1063 / 1045 / 1056 / 1055 | 7.6% | 70% / 64% | 边缘，与修前 7.9% 持平 |
| 同上，4×24 | 96 | 1346 / 1758 / 1473 / 1271 | 29% | 72% / 77% | 不达标 |

Token2Wav 不再是绑定 stage，绑定的是 LLM 卡自己的每帧延迟链。把 4×20 里一个进程的每帧时间拆开看：编码器（在编码器卡上，含合批等待）p50 171 ms、p95 223 ms；prefill 从提交到拿到结果 350 到 450 ms，其中凑批等待平均 126 ms、一次 forward 155 ms；decode p50 126 ms（listen 帧 3 个 token，speak 帧平均 7 个，每个 token 约 36 ms）；其余是主机侧准备与回传。加起来就是 p50 780 ms，speak 帧多几个 token 就是 p95 的 960 ms。主机不是瓶颈：节点 384 核，负载 8。

prefill 那 350 到 450 ms 里有一个结构性的浪费：编码完成的散开（p95 223 ms）超过了 200 ms 的凑批 deadline，所以每个 tick 的 prefill 实际被拆成宽度 13 左右的两批，第二批要等第一批算完才能开始。

编码器那 171 ms 也不是算力：一批 4 张图的 GPU 时间 60 ms，每张 15 ms；慢在每个进程的 20 张图落到 2 张编码器卡上，每卡 10 张要排 2 到 3 批。为了缩短它试了两条路，都不成：

| 试法 | session | P95 (ms) | 超时帧 | 说明 |
|---|---|---|---|---|
| 视觉合批上限 4 改 8，4×22 | 88 | 1071 / 1106 / 1033 / 1057 | 9.2% | 批 8 的 GPU 时间约是批 4 的两倍，说明视觉编码已经是算力而非 launch 开销，合大批没有收益 |
| 编码器副本同时放到 2 张 Token2Wav 卡上（4 张编码器卡），4×22 | 88 | 3678 / 2849 / 2069 / 3214 | 47% | Token2Wav 卡 95% 到 97%，放上去的编码器副本被饿死，视觉编码 p95 从 223 ms 涨到 1254 ms |
| 5 LLM + 3 张「编码器 + Token2Wav」混合卡，5×20 | 100 | 3544 / 3924 / 2690 / 2832 / 3240 | 81% | 三张混合卡 96% 到 98%，prefill 合批宽度掉到 7.5；编码器与 Token2Wav 同卡这条路第三次证伪 |

所以 8 张卡的分法仍然是 4 LLM + 2 编码器 + 2 Token2Wav，80 个 session 达标、88 边缘。再往上要在 LLM 进程的延迟链上省出 100 到 150 ms，才能让每卡 22 到 24 个 session 达标（88 到 96）。

### 在延迟链上省出来的：prefill 凑批参数

编码完成散开超过 deadline 导致 prefill 拆成两批、第二批等第一批，这个结构说明「凑满再放」不如「小批快放」。把 `OMNI_EXEC_PREFILL_TARGET` 从 N 降到 8、`OMNI_EXEC_PREFILL_DEADLINE_US` 从 200 ms 降到 60 ms，每个 tick 变成 3 批宽度 8 左右的 prefill，各 112 ms，第一批在 60 ms 内就发出：

| 布局（4 LLM + 2 编码器 + 2 Token2Wav，错峰，barrier 同起） | session | 各进程 P95 (ms) | 超时帧 | 对照（target N / 200 ms） |
|---|---|---|---|---|
| 4×20，target 8 / 60 ms | 80 | 897 / 876 / 948 / 933 | 1.7% | 3.5% |
| 4×21，target 8 / 60 ms | 84 | 967 / 929 / 1029 / 973 | 3.5% | 未测 |
| 4×22，target 8 / 60 ms（两台各一次） | 88 | 1134 / 1009 / 979 / 988；1047 / 960 / 1060 / 995 | 6.5%、5.7% | 7.6% |
| 4×22，target 11 / 100 ms | 88 | 1040 / 1097 / 1125 / 1165 | 9.6% | 更差：prefill 批更多但仍等，decode 被堵 |

p50 稳定降 65 ms（835 到 770），超时率降 2 个百分点左右，可复现。80 个 session 从「达标」变成「余量明显」（P95 全部低于 950 ms），84 达标，88 仍是边缘。这是不改代码能拿到的全部。

### 视觉编码器与 Token2Wav 的 kernel 构成（nsys，8 session 单进程，按卡拆）

视觉编码器卡上一批 4 张图 55 ms 的 GPU 时间：注意力的 softmax 22%（每次 439 us，是 1024×1024×16 头的 f32 softmax，纯访存）、cuBLAS F16 GEMM 约 35%、f32 与 f16 之间的来回转换 13%、偏置加法 8%、拷贝 5%。把线性层权重换成 Q8_0（MMQ 路径）后总时间不变（4042 对 3980 ms），MMQ 与 cuBLAS F16 在这个形状上打平，所以量化视觉权重没有收益。有收益的是把注意力换成 flash attention：softmax 和两次注意力矩阵乘、连带的转换一起消掉，预计每张图 15 ms 降到 11 ms 上下，编码器 wall 171 ms 降 40 ms 左右。

Token2Wav 卡上每个 chunk 55 ms 的 GPU 时间是 launch 堆出来的：8 个 session 36 秒里 140 万次 kernel 启动，每个 chunk 约 6000 次，排前面的是 3 us 的逐元素加法（46 万次）、拷贝（18 万次）、im2col、concat；单个最重的是声码器的转置卷积 `conv_transpose_1d`，一次 4.2 ms，每 chunk 两次占 15%。这一段适合 CUDA graph 和一个像样的转置卷积 kernel，预期 2 倍以上。Token2Wav 现在已经不在关键路径上，但如果它每卡能撑 90 个 session，8 卡里就能只留 1 张给它，腾出的那张做第三张编码器卡，编码器 wall 直接少一批。

主干 LLM 卡（batched-bench 真实形状 79 token × 22 seq，单卡）：prefill forward 337 ms 里 MoE 的 MMQ kernel 占 50%、其它 MMQ 与 quantize/get_rows/ids 辅助 15%、gated delta net 15%，MoE 部分有效算力约 48 TOPS，是 H20 int8 峰值的六分之一。decode 22 宽 33 ms 里 MMQ 53%、递归状态逐层拷贝 15%（每步每层 `get_rows` 搬 62.8 MiB/seq 的状态）、MMVQ 10%。这两条是最大的杠杆，也是最重的活：MoE 分组 GEMM 是 kernel 项目，递归状态不拷贝是 llama.cpp 图层面的改动。

### 当前结论（2026-09-05）

8 卡 H20、Q8_0 权重、F16 KV、每 session 每秒 1 帧音频 + 1 帧图：4 LLM 卡 + 2 编码器卡 + 2 Token2Wav 卡，80 个 session 达标且有余量，84 达标，88 边缘。再往上的杠杆按预期收益排：MoE prefill/decode kernel（2 到 3 倍空间，重）、视觉编码器 flash attention（每张图约 −4 ms，中）、Token2Wav 的 CUDA graph 与转置卷积 kernel（每卡容量 2 倍以上，中；间接换出一张编码器卡）、decode 里的递归状态拷贝（15%，中）。

## 2026-09-05 下午（续）：视觉编码器 flash attention（排序第 1 步）

改动在 vision.cpp 的注意力构图：QK^T、softmax、V 乘三步换成 `ggml_flash_attn_ext`，K/V 转 F16，merger 层的窗口 mask 转 F16。第一版没有收益：SigLIP 的 head 维是 72（1152/16），ggml 的 CUDA flash attention 把 72 明确排除在 tensor core（mma）kernel 之外，落到通用 tile kernel，一次 610 us，和原来 softmax 加两次矩阵乘打平。把 head 维零填充到 80（mma 支持的尺寸，QK^T 不受零维影响，V 乘多出的零列算完切掉）之后 mma kernel 一次 209 us。顺手把线性层的 cuBLAS 输出改成直接写 F32（`ggml_mul_mat_set_prec`），去掉了每个线性层后面那次 F16 到 F32 的转换 kernel。

| 视觉编码器（8 session 单进程，nsys 按卡拆） | 编码器卡 kernel 总时间 (36 s) | 一批 4 张图 GPU 时间 | 4×22 布局下编码器 wall p50 / p95 |
|---|---|---|---|
| 原路径（mul_mat + soft_max） | 3980 ms | 61 ms | 157 / 215 ms |
| FA，head 72（tile kernel） | 3949 ms | 60 ms | 未测 |
| FA，head 填充到 80（mma kernel）+ F32 GEMM 输出 | 3021 ms | 46 ms | 130 / 175 ms |

数值检查：同一张图 FA 开关前后的输出 embedding（66 token × 2048）余弦相似度 0.999988，最大绝对偏差 0.014（最大值 1.94 的 0.7%），平均偏差 1.8e-4，在 F16 精度范围内。开关：`OMNI_VISION_FA=0` 回退，`OMNI_VISION_FA_PAD` 改填充尺寸，`OMNI_VISION_MM_F32=0` 回退 GEMM 精度。

编码器快了 24%，但 4×22 的超时率没有动（5.4% 到 6.4%，与 5.7% 到 6.5% 的对照在噪声之内）：每帧 760 ms 里编码器只省了 27 ms，尾部由 LLM 卡决定（4 张 LLM 卡 78% 到 96%）。

顺带测了两种依赖「Token2Wav 只占 1 张卡」的布局，都因 Token2Wav 卡显存不足而崩：4×22 全部 Token2Wav 放 1 张卡，以及 5 LLM + 2 编码器 + 1 Token2Wav 的 5×18。原因是每 session 的 Token2Wav 私有 backend 约占 1.1 GB（44 个 session 时 48 到 52 GB），88 到 90 个放不进 96 GB。活下来的那个 5×18 进程 P95 838 ms，说明 LLM 卡每张 18 个是很松的。所以第 2 步（Token2Wav）的目标变成两件事：每 session 显存降到 1 GB 以下，且单卡算力撑 90 个 session（现在 44 个用 70% 到 77%）。两件都做到，8 卡才能改成 5 张 LLM 卡，上限到 90 到 100。

## 2026-09-05 下午（续二）：Token2Wav 的真实构成，第 2 步的前提变了

第 2 步原计划是「给 Token2Wav 加 CUDA graph，换转置卷积 kernel」。动手前先按卡、按 CUDA graph 粒度重新剖了一次（nsys `--cuda-graph-trace=graph`，8 session 单进程），发现前提不成立：token2mel（conformer 编码器 + 5 步 flow matching）的 session 图本来就是预建的，ggml 的 CUDA 后端已经把它捕获成 CUDA graph 在复用，36 秒里 265 次 graph 启动、GPU 时间 11.1 s，平均每个 chunk 46 ms。也就是说那 6000 个 kernel 不是主机 launch 开销，是 GPU 上逐个执行的小 kernel（平均 7.7 us 一个，占不满 SM）。声码器每次调用重建图，没有被捕获，每 chunk 14 ms，其中转置卷积 4.5 ms、im2col 1.9 ms。

显存也量清了：每个 session 在 Token2Wav 卡上约 1.0 GB，其中 token2mel 的 session 分配器一次预留 808 MiB（两张预建图加持久 cache 的激活空间），声码器私有 backend 约 120 MiB。这就是 88 到 90 个 session 放不进一张卡的原因。

Token2Wav 单卡的真实容量也比之前估的低。两张 Token2Wav 卡各 44 个 session 时（4×22），Token2Wav 线程看到的每 chunk 推理时延 p50 407 ms、p90 944 ms，排队 p90 702 ms；各 40 个时（4×20）p50 222、p90 709。多 session 在各自 stream 上并发只能把小 kernel 叠到一定程度，40 个之后就叠不动了。

所以「让 Token2Wav 只占 1 张卡、腾出第 5 张 LLM 卡」需要 Token2Wav 单卡撑 90 个 session，等于算力效率 2.5 倍以上、显存每 session 降到 0.5 GB 以下。这两件事指向同一个改动：把 token2mel 做成跨 session 合批（flow matching 的 CFG batch 从 2 扩到 2N，各 session 的 prompt 与 cache 堆叠、按长度补齐；分配器按 worker 而不是按 session 预留）。小 kernel 的 GPU 时间几乎不随 batch 增长，合批 8 个 session 预计每 chunk 从 46 ms 降到 10 ms 以内；再加转置卷积 kernel，每 chunk 从 60 ms 到 20 ms 以内。这是三到五天的改动，不是原先估的一到两天。转置卷积 kernel 单独做只能省 13%，不足以改变布局。

## 2026-09-05 下午（续三）：MoE kernel（排序第 2 条）第一刀，88 达标

读完 ggml 的 MMQ（量化矩阵乘）代码，MoE 的 mul_mat_id 路径上有两个结构性浪费，都与 kernel 内的 mma 无关：

1. tile 网格按「每个 expert 都拥有全部列」枚举。1738 个 token 乘 8 个 expert 是 13904 列，按 128 宽切成 109 个列 tile，网格是 109 × 256 个 expert × 行 tile 数，其中每个 expert 只有 1 个列 tile 有活（平均 54 列），其余的 block 启动后发现越界就退出。gate/up 矩阵每次启动约 11 万个 block、down 矩阵 45 万个，有活的只有 1024 和 4096 个。
2. tile 宽度按总列数 13904 选成最大的 128，而单个 expert 平均只有 54 列，mma 算力有 58% 花在填充上。

改法（ggml-cuda/mmq.cu、mmq.cuh，约 130 行）：一个单 block 的小 kernel 对各 expert 的 ceil(列数 / tile 宽) 做前缀和得到紧凑的 tile 起点表，主 kernel 里每个 block 用二分查找把紧凑 tile 号映射回（expert，expert 内 tile 号），网格上限变成「总列数 / tile 宽 + expert 数」；tile 宽度按每 expert 的平均列数选（`GGML_CUDA_MMQ_ID_X` 可强制，`GGML_CUDA_MMQ_ID_COMPACT=0` 回退旧路径）。stream-k 的 K 切分与 fixup kernel 用同一套映射。中间踩过一个坑：mul_mat_id 传给 MMQ 的「最大列数」是 token 数而不是（token, expert）总列数，紧凑网格的上限一度按 token 数算，expert 有两个以上 tile 时后面的 tile 没人算，test-backend-ops 的 769 个 MUL_MAT_ID 用例里 145 个失败；改成按总列数之后全部通过。

| MUL_MAT_ID，Q8_0，256 expert，top-8（test-backend-ops perf，H20 单卡） | 旧路径 | 紧凑 tile（自动宽度） | 倍数 |
|---|---|---|---|
| gate/up，1738 token（512 × 2048） | 1028 us | 735 us | 1.40 |
| down，1738 token（2048 × 512） | 1616 us | 810 us | 2.00 |
| gate/up，632 token（8 session） | 808 us | 423 us | 1.91 |
| gate/up，22 token（decode） | 163 us | 147 us | 1.11 |

batched-bench 真实形状（79 token × 22 seq）的 prefill forward 从 327 ms 降到 284 ms（5308 到 6121 token/s），decode 不变。8 卡布局的效果：

| 4 LLM + 2 编码器 + 2 Token2Wav，prefill target 8 / 60 ms，视觉 FA | session | 各进程 P95 (ms) | 超时帧 | 改前 |
|---|---|---|---|---|
| 4×22 | 88 | 903 / 917 / 937 / 965 | 2.9% | 5.4% 到 6.5% |
| 4×24 | 96 | 1165 / 1164 / 1082 / 1167 | 9.7% | 29% |

**88 达标**（P95 全部低于 1 s），96 从不达标变成边缘。

decode 那边另试了一刀：22 个 token 的 mul_mat_id 走的是 tile 宽 8 的 MMQ，而每个活跃 expert 平均只有 1.4 列，向量 kernel（MMVQ）更合适，但它有每次最多 8 个 token 的硬限制；把 token 切成 3 段各跑一次向量 kernel（`GGML_CUDA_MMVQ_MMID_CHUNK_MAX`），微基准上 148 到 103 us、205 到 143 us，正确性通过，但 batched-bench 的 decode 吞吐没有变化，原因待查。

复核（两台节点、同一配置）：4×22 在另一台上 2.6%（P95 903 到 943 ms），可复现；4×23 = 92 为 4.3%（P95 909 到 1042 ms），边缘。**当前 8 卡结论：88 达标，92 边缘，96 不达标（9.7%）。**

decode 侧的 kernel 构成（batched-bench 22 宽，每次 forward 33 ms，nsys）：MoE 的 MMQ 11 ms（紧凑 tile 之后，读权重的有效带宽约 1.5 TB/s，是 HBM 峰值的 38%）；递归状态逐层拷贝（`k_get_rows_float`）4.4 ms；gated delta net 的 qkv 投影 2.9 ms——它在 decode 时的输入张量是 [2048, 1, 22]（token 在第三维），CUDA 后端按 22 个 channel 各做一次矩阵向量乘，16.8 MB 的权重重复读 22 遍；dense 投影的 MMQ 4 ms；GDN 本体 1.4 ms。切 token 分段走向量 kernel 那一刀在真实模型上反而慢（33 us × 3 段比紧凑 MMQ 的 91 us 还多），因为真实的 expert 分布偏斜、活跃 expert 少，已关闭（默认 0）。

### decode 侧两刀（2026-09-05 傍晚）

两处都不是 kernel 本身，是张量布局和一个拷贝 kernel 的写法：

1. gated delta net 层的输出投影。图里把注意力输出整成 [4096, n_seq_tokens, n_seqs] 再乘 `ssm_out`，多 session decode 时 n_seq_tokens = 1、n_seqs = 22，CUDA 后端把第三维当 batch，对 22 个 sequence 各做一次矩阵向量乘，8 MB 权重重复读 22 遍，每层 101 us、每次 forward 2.9 ms。改成 [4096, n_tokens] 的二维布局（内存顺序不变，结果张量与原先相同），变成一次 22 列的 MMQ。
2. 递归状态的逐层 gather（`k_get_rows_float`）。每层每步把 22 个 sequence 各 2 MB 的状态用 get_rows 搬一遍，原 kernel 每线程搬 4 字节，只跑到 0.6 TB/s，每次 forward 4.4 ms。加了一个 float4 向量化的变体（行长与步长都是 4 的倍数、行长不小于 4096 时启用），GET_ROWS 的 47 个用例全过。

| batched-bench，单卡 H20，79 token × 22 seq | prefill forward | decode（22 宽） |
|---|---|---|
| 今天早上（原 MMQ） | 327 ms（5308 t/s） | 654 t/s（33 ms/步） |
| 紧凑 expert tile | 284 ms（6121 t/s） | 657 t/s |
| 再加两刀 decode | 280 ms（6199 t/s） | 799 到 825 t/s（27 ms/步） |

decode 每步快 22% 到 25%。

8 卡布局（4 LLM + 2 编码器 + 2 Token2Wav，prefill target 8 / 60 ms，视觉 FA，紧凑 expert tile，decode 两刀）：

| session | 各进程 P95 (ms) | 超时帧 | 今天早上 | 结论 |
|---|---|---|---|---|
| 4×22 = 88 | 900 / 853 / 874 / 914 | 0.5% | 7.6% | 达标，余量明显 |
| 4×23 = 92 | 910 / 912 / 909 / 1003 | 3.1% | 未测 | 达标 |
| 4×24 = 96（两台） | 1077 / 946 / 942 / 1059；1169 / — / 963 / 969 | 4.6%、约 6% | 29% | 边缘 |

**2026-09-05 收盘：8 卡 H20 上 92 个 omni session 达标，96 边缘。** 今天早上是 80 达标、88 边缘。

decode 每步还剩 27 ms：MoE 的 MMQ 约 11 ms（读权重的有效带宽只有 HBM 的 38%，需要按 expert 分组的矩阵向量 kernel），dense 投影 MMQ 约 4 ms，状态 gather 向量化后约 1.5 ms，GDN 本体 1.4 ms。prefill forward 280 ms 里 MoE 的 MMQ 仍占约 40%（mma 有效算力约 40 TOPS，int8 峰值的 13%），再往下是 MMQ kernel 本身在 Hopper 上的效率（没有用 wgmma），属于更重的 kernel 工作。

## 2026-09-05 晚：目标改为 400 个 session，开始做 expert parallel（EP）

400 个 session 在 8 卡上意味着每秒 400 帧：prefill 31600 token/s，decode 约 2000 token/s，400 张图，约 140 个音频 chunk。按今天的效率折算成卡数：视觉编码 5 到 6 张，Token2Wav 8 张以上，LLM 7 到 8 张，合计 20 张以上。三个 stage 都要提效两到四倍才能塞进 8 张卡；显存侧则要求 LLM 权重不再每卡一份，这就是 EP。

EP 收益先量了一遍（test-backend-ops，模拟每卡持有 32 个 expert、收到 8 张卡的 token）：prefill 的 MoE 矩阵乘从 735 us 到 395 us（1.9 倍，tile 填满了），decode 从 132 到 91 us 但会被每层两次 all-to-all 吃掉。端到端预期：prefill forward 快约 19%，decode 持平，显存上限从约 820 到约 1100。

分三步做：

1. ggml CUDA 后端新增「expert 切分」buffer 类型（`ggml-cuda/esplit.cu`）：三维 expert 权重按 expert 序号切成连续段放到多张卡上，只保留一份。这种权重上的 mul_mat_id 在调用方那张卡上完成 (token, expert) 列的按 expert 排序和量化，每张 owner 卡用一个 gather kernel 经 NVLink 把自己那些 expert 的列、目标行索引和 expert 边界拷到本地，跑本地的分组 MMQ（复用今天的紧凑 tile 代码，expert 边界指针偏移到本卡的 expert 段），结果通过 peer 访问直接写进调用方卡的输出张量。GPU 侧用 event 串接，没有主机同步。loader 侧用 `LLAMA_EXPERT_SPLIT_DEVICES=0,1,2,3` 把 `*_exps` 张量放进这种 buffer。
2. 多个 session 组（每张 LLM 卡一个上下文，dense 权重各一份、共享同一份切分 expert）在每个 expert 算子处汇合，把各自的 token 合成一次分组 GEMM，这才拿到 1.9 倍。
3. 全局步进调度：所有 LLM 卡同一步 forward，prefill 对齐到同一步，没有 token 的卡也参与算自己那份 expert。

第 1 步单独就能验证正确性、跨卡开销和显存分布；第 2、3 步才有算力收益。

第 1 步落地（2026-09-05 晚）。踩的坑：新文件要重新 configure 才进 CMake 的源文件通配；owner 卡的 kernel 直接读主卡的临时缓冲时报非法访问，原因是 ggml 的 CUDA 内存池用的是 VMM（虚拟内存映射）分配，这类映射不受 `cudaDeviceEnablePeerAccess` 管，必须在映射时用 `cuMemSetAccess` 把对端设备也加进访问列表（加了 `GGML_CUDA_VMM_PEER=1`）。修完之后：

| 单进程，dense 在 1 张卡，expert 切到 4 张卡（`LLAMA_EXPERT_SPLIT_DEVICES=0,1,2,3`） | prefill forward（79 token × 22 seq） | decode（22 宽） |
|---|---|---|
| 单卡（今天下午的基线） | 286 ms（6078 t/s） | 773 t/s |
| EP 4 卡 | 231 ms（7519 t/s） | 770 t/s |

贪心生成 48 个 token 与单卡逐字相同。prefill 快 19%，来自 MoE 部分被 4 张卡分摊；decode 持平（省的算力被每层两次跨卡传输抵消）。这一步还没有显存收益，因为每个进程仍各自持有一份切分的 expert；显存收益要等第 3 步（多个 session 组共享同一份 expert）。

### 2026-09-06 凌晨：单进程多模型共享 expert 跑通，未协调的 EP 在 omni 上慢 2 倍

多进程 EP 不可用（4×24 每帧 p50 25 s）：不同进程在同一张 GPU 上是不同 CUDA context，只能按时间片切换，EP 让每次 forward 的 120 个 expert 算子都要等 4 张卡回应，每一步都撞上别的进程的时间片。

于是改成单进程：expert 切分 buffer 里加了按张量名的注册表（`GGML_CUDA_ESPLIT_SHARE=1`），第二个及之后的模型实例直接引用第一个实例已放好的 expert 分片，不再分配、不再上传；omni 的多工作者模式改成每张 LLM 卡一个模型实例（dense 权重在本卡、expert 共享）各配一个执行器（`OMNI_EXEC_WORKERS=4`，`OMNI_EXEC_WORKER_DEVICES=CUDA0,…`）。bench 加了 `--stagger-groups`（进程内把 session 的帧节拍分成几组错开）和 `--begin-threads`（session_begin 每个约 14 s，96 个串行要 22 分钟）。

单进程 96 个 session、4 个工作者的结果：显存达到预期，LLM 卡 1 到 3 各 20 GB（原来 56 GB），卡 0 因为还带着 TTS 执行器和编码器根副本是 52 GB。但延迟不行：每帧 p50 6.8 s，超时 86%；decode 一次 forward 59 ms（原 27 ms），prefill 139 ms（持平），每个工作者的执行器忙 75%。原因是 4 个工作者各自独立地 forward，每个 expert 算子都要等 4 张卡各自算完自己那份，而每张卡上同时穿插着另外三个工作者的 expert kernel 和 dense kernel，随机排队把每一层的等待都拉长，120 层下来 decode 慢一倍。这就是之前说的：没有汇合合批的 EP 只会更慢。

下一步是第 2 步：4 个工作者在每个 expert 算子处汇合，由最后到达者对 4 张卡各发一次「所有工作者 token 的并集」分组 GEMM，写进本地暂存再散射回各工作者的输出；执行器改成全局按步推进（有 token 的工作者同一步一起 forward，参与者数量在发起前公布）。这一步同时拿到 1.9 倍的 kernel 效率和确定的等待，而不是随机排队。

### 2026-09-06 凌晨：第 2 步跑通，4 个工作者在 expert 算子处汇合合批

实现：CUDA 后端里每个 expert 张量一个汇合点，K 个参与者到齐后由最后到达者对每张 owner 卡发一组 kernel：算各参与者在本卡 expert 上的列数与偏移、把并集的列按 expert 再按参与者归并到一块 [K 块][列] 的量化缓冲、跑一次紧凑 tile 的 MMQ 写到本地暂存、再按（参与者，目标列）散射回各自的输出张量，全程用 event 串接，主机不同步。执行器侧加了全局步进：有成熟工作的执行器发起一轮并定下这一轮的步长（prefill 79 或 decode 1），所有执行器只拿这一长度的步，最后到达者把参与者数量公布给 CUDA 后端；步长统一是为了每个参与者的一次 llama_decode 只有一个 ubatch，否则汇合次数对不上会死锁。

踩的坑：ggml 的 VMM 内存池要求按后进先出释放，分多个容器交错释放会断言，改成每张卡一块内存内部切分；TTS 的执行器也走进了步进分支，barrier 计数错乱，一个参与者等第二个等到超时，改成只让注册过的 LLM 执行器参与。

32 个 session（4 工作者 × 8）通过：p50 532 ms、p95 1131 ms、超时 9.1%、无失败帧，speak/listen 计数正常；1152 轮里参与者为 1、2、3、4 的分别 330、264、221、109 轮。代价已经能看到：decode 步要等轮次，平均排队 42 ms；这个规模下每卡只有 8 个 session，合批收益还没出来，要看 96。

96 个 session（4 工作者 × 24，lock-step EP，同一进程）：p50 2.7 s、超时 93%，对照的多进程非 EP 4×24 是 4.6%。1224 轮里 451 轮是 4 个参与者合批，合批本身是成立的，慢在三处：

1. 汇合后的 4 张卡的 kernel 全由最后到达者一个线程发起：每个 expert 算子每张卡约 5 个 kernel 加 event，4 张卡 20 次启动、约 100 us 主机时间，120 个算子就是每次 forward 约 12 ms 的串行主机开销。非 EP 时每个工作者在自己的线程上各发 3 个 kernel，是并行的。decode 一次 forward 32 ms（非 EP 27 ms），并没有因为 MoE kernel 变快而变短。
2. 合并前后多了 gather 和 scatter 两遍数据搬运和一次边界计算，decode 时这三个 kernel 的时间比合并后的 GEMM 本身还长。
3. 轮次耦合把批做宽了：一个工作者的 prefill 成熟，四个工作者都把手里的 prefill 倒进这一轮，每卡 prefill 宽度从 5.7 到 10.7，一次 forward 从 139 ms 到 220 ms，同一轮里其它工作者的 decode 步全在等（prefill 排队均值 163 ms、decode 排队 34 ms）。合并后的 prefill 吞吐反而更低：四卡合起来每秒 15.4k token，非 EP 四卡独立是 22.6k。

显存收益如预期（LLM 卡 20 GB 对 56 GB），但 400 个 session 卡的是算力。要让 EP 在这个负载上不亏，至少还要：各参与者并行为各自的卡发 kernel（每个算子再加一次 barrier）、把边界 / gather / scatter 融成一个 kernel、轮次调度只在 prefill 时合批而不把 decode 拖进来。即便都做到，理论上限也只是 prefill 快 19%、decode 持平。结论：在这个延迟约束的 omni 负载上，EP 是显存工具而不是算力工具；提升算力上限的路仍然是单卡 kernel（MoE decode 的分组矩阵向量乘、MMQ 在 Hopper 上的 mma 效率）和三个 stage 的效率。

## 2026-09-06 上午：Token2Wav 显存修掉 —— 每 session 只留 cache 快照（slot 模式）

问题拆清楚了：每个 session 的 token2mel 分配器预留 808 MiB，其中真正持久的是 4 个 stream cache（conformer cnn 12 KiB / conformer att 12 MB / estimator cnn 1.3 MB / **estimator att 193 MB**，共约 206 MB），剩下约 600 MB 是每 chunk 的激活峰值，只在每秒几十毫秒里用到。跨 session 合批本来就要把 N 个 session 的 chunk 堆到一张图、cache 拷进拷出，所以这里直接改成「每卡一个槽位池」：每个槽位一套预建图 + 激活空间 + 声码器 backend（`OMNI_T2W_SLOTS` 张卡内固定数量），session 只持有 4 个 cache 的设备快照，每个 chunk 借一个槽位算出结果。

| 单张 Token2Wav 卡 | 每 session 显存 | chunk 推理 p50 / p90 | 结论 |
|---|---|---|---|
| 24 session，经典（每 session 独占） | 1.0 GB | 103 / 227 ms | 基线 |
| 24 session，8 槽位 | 0.65 GB | 93 / 176 ms | 无回退 |
| 48 session，8 槽位 | 0.42 GB | 100 / 198 ms，排队 0 | 单卡 48 个仍有余量；经典路径 44 个已 87% 忙、48 个直接 OOM |

按 4×24 布局换算：每张 Token2Wav 卡 48 个 session，每 session 约 0.35 到 0.45 GB，两张卡峰值各约 21 GiB（原 44 个 session 时已到 48 到 52 GiB 上限）。96 个 session 的 4×24 布局跑通，T2W 卡不再撞显存上限（原来是 88 到 90 就 OOM）。

一个正确性问题：slot 模式每 chunk 要把 cache 从槽位拷回 session 快照，拷贝走的是每线程流，和槽位自己的计算流之间没有排序，可能与图计算竞争（经典路径 cache 常驻图内、每 chunk 不回拷，所以没这个竞争）。已在拷回前 `ggml_backend_synchronize` 槽位 backend。
