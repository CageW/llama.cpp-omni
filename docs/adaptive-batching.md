# 自适应合批：让 executor 自己决定何时发 forward

2026-09-18。分支 `feature/adaptive-batch-scheduler`，基线 `feature/server-multisession`。
测试机：南京 H20 节点 `vm-7-13-ubuntu`，权重 MiniCPM-o-4.5 F16。压测工具 loadsim 在分支
`perf/realtime-load-simulator-adaptive`。

## 问题

多 session 时所有 session 的 LLM forward 走同一个共享 executor：每个 session 把自己的下一步
（一次 prefill 或一个 token 的 decode）投进队列，worker 线程把队列里的步合成一次 `llama_decode`。
合批的收益很陡：在 H20 上一次 1 token 的 forward 单独跑 6.8 ms，8 路合在一起 9.3 ms，16 路
11.7 ms，32 路 15.9 ms。也就是说宽度 16 时每路只摊到 0.7 ms，比单独跑便宜十倍。

但 worker「什么时候发」是几个启动时读死的参数决定的：等待窗口 3 ms，prefill 凑批目标数，
prefill 最长等待 200 ms。这套参数在两头都不对。负载低时窗口是纯粹的额外延迟；负载中等时
各 session 的步不是同时到的（每个 session 拿到结果后要在主机侧采样约 6 ms 才投下一步，
而各 session 的相位随机），3 ms 的窗口等不到别人，一轮本可合成一次的 forward 被拆成三四次
小的，GPU 时间花在固定开销上。负载高到队列永远非空时倒是自然合批了，但那时延迟已经超标。

benches 里之前的结论是「多等一会儿凑批的所有参数都更差」，原因是 decode 和 prefill 在同一个
队列里，凑 prefill 会把 decode 一起扣住。后来的 `ripe` 策略只扣 prefill 不扣 decode，把
单卡上限从 16 抬到 24，但它仍是固定阈值加固定 deadline，而且 decode 那一侧完全不合批。

## 做法

`OMNI_EXEC_POLICY=adaptive`。worker 在每次把结果交还给某个 sequence 时记下时刻，在该 sequence
下一次投步时算出「交还到再投」的间隔；只有间隔短于 60 ms 的算链内间隔（长于这个数说明它在等
客户端的下一帧，不是在采样），对每个 sequence 维护这个间隔的均值和平均绝对偏差。

队列非空时 worker 不立刻发，而是先看：哪些 sequence 刚拿到结果、还没投回来、按它自己的间隔
均值加两倍偏差算还没到期。这些是「马上会回来的」。等它们回来能省下的是 k 次 forward 的固定
开销；GPU 空闲时省下的 GPU 时间没有价值，所以按 GPU 忙占比加权。等待的代价是已经排队的步
多等 Δ。规则是 Δ 小于 k 乘固定开销乘忙占比乘增益（默认 1.5）时才等，等到第一个预测到达就
重新评估，任何一个已排队的步总共最多被扣 20 ms。GPU 忙占比是 forward 时间除以墙钟时间的
指数滑动平均，固定开销从单步单 token 的 forward 时间在线估计。

这和 Symphony（2023）的 deferred batching 是一个思路：按成本曲线和 deadline 算出等待边界，
而不是固定窗口。区别是这里「下一个到达」不是泊松假设，是每个 sequence 自己的确定性回投时间。
文献里没查到有人在 LLM decode 上做这一层预测；Metronome（2026）做的是按帧节拍 lock-step，
没有处理一帧内多步 decode 的相位问题。

## 服务时间曲线

工具 `llama-omni-perf-exec-cost` 走服务端同一条 executor 路径，让 W 个线程同时投 W 个相同的
步，测合并后一次 forward 的墙钟时间。KV unified，n_ubatch 512。

| 每步 token 数 | 宽 1 | 宽 4 | 宽 8 | 宽 16 | 宽 24 | 宽 32 |
|---|---|---|---|---|---|---|
| 1（decode） | 6.8 ms | 8.7 | 9.3 | 11.7 | 14.8 | 15.9 |
| 5 | 8.1 | 13.2 | 14.3 | 22.3 | 23.3 | 33.5 |
| 12（一帧音频的 prefill） | 9.6 | 14.0 | 22.0 | 33.4 | 47.1 | 60.2 |
| 80（omni 一帧） | 20.7 | 48.8 | 97.9 | 195 | 292 | 374 |

decode 合批几乎免费，prefill 合到 4 路以后每路 12 ms 线性增长（MoE 算力墙，与 benches 一致）。
prefill 和 decode 混进同一个 forward，在 KV unified 下只比纯 prefill 多 1 到 4 ms，所以策略
不需要把两类分开，unified KV 是前提。

上面是 KV 位置 0 的数字。把每个 sequence 先填到位置 800 再测（`--pos 800`，q8_0 KV，4096
上下文），也就是几十路 session 各聊了十几分钟之后的真实形状：

| 形状 | 宽 1 | 宽 8 | 宽 16 | 宽 32 |
|---|---|---|---|---|
| 2 token decode（listen 帧） | 17.1 ms | 41.4 | 57.3 | 82.4 |
| 同上，再加同样宽度的 11 token prefill 混在一个 forward | 37.7 | 51.6 | 173.8 | 502.5 |

位置 800 时单步 decode 是位置 0 的 2.5 倍，32 路混合 forward 半秒。attention 读 KV 的开销
随位置线性涨，合批能省的固定开销在总成本里的占比随之缩小。

## 端到端对比

LLM 与 TTS 各一个共享 executor 在 2 号卡，编码器在 3 号卡，Token2Wav 独占 4 号卡。loadsim 在节点内直连，每帧 1000 ms（与产品前端一致），trickle 到达 15 秒内铺开，每
session 固定 70 秒，整场 95 秒，随机种子固定，system prompt 与 `length_penalty` 用产品默认。
判据：决策延迟（从发出第 k 帧到收到模型对第 k 帧的 listen 或 text）p95 不超过 1 秒。

每个并发点都重启服务再测，两种策略同一随机种子。`w` 是实际合批宽度（平均每次 forward 合了
几个 step），`busy` 是 LLM 卡上 forward 时间占墙钟的比例，`hold` 是自适应策略主动扣住队列的
总时长。

### 24 槽，每 sequence 8192 上下文，F16 KV

| 并发 | 贪心 p95 | 贪心 w | 贪心 busy | 自适应 p95 | 自适应 w | 自适应 busy | hold |
|---|---|---|---|---|---|---|---|
| 12 | 0.075 s | 1.12 | 0.24 | 0.072 s | 1.01 | 0.26 | 65 ms |
| 16 | 0.084 s | 1.08 | 0.34 | 0.076 s | 1.17 | 0.32 | 341 ms |
| 20 | 0.098 s | 1.30 | 0.38 | 0.074 s | 1.20 | 0.41 | 470 ms |
| 24 | 0.095 s | 1.27 | 0.47 | 0.082 s | 1.34 | 0.45 | 830 ms |

两种策略都远在预算之内。这个档位的上限是显存：24 槽 F16 KV 在 LLM 卡上占 57 GB，40 槽
占 84 GB。

### 96 槽，每 sequence 4096 上下文，q8_0 KV

| 并发 | 贪心 p95 | 贪心 w | 贪心 busy | 自适应 p95 | 自适应 w | 自适应 busy | hold |
|---|---|---|---|---|---|---|---|
| 40 | 0.169 s | 2.21 | 0.74 | 0.153 s | 2.34 | 0.70 | 1.5 s |
| 48 | 0.187 s | 2.93 | 0.76 | 0.175 s | 3.04 | 0.74 | 1.7 s |
| 56 | 0.214 s | 3.68 | 0.80 | 0.213 s | 3.79 | 0.78 | 1.9 s |
| 64 | 0.317 s | 4.81 | 0.79 | 0.324 s | 5.07 | 0.78 | 1.5 s |
| 72 | 0.348 s | 6.26 | 0.81 | 0.332 s | 6.33 | 0.81 | 1.7 s |
| 80 | 0.505 s | 7.81 | 0.84 | 1.084 s | 9.18 | 0.84 | 1.4 s |
| 88 | 1.189 s | 11.85 | 0.83 | 1.354 s | 13.22 | 0.83 | 1.2 s |
| 96 | 4.258 s | 20.13 | 0.85 | 3.873 s | 20.13 | 0.84 | 0.8 s |

80 路那一行两种策略差了一倍，所以在 80 和 84 各换两个随机种子重跑：

| 并发 | 种子 | 贪心 p95 | 自适应 p95 |
|---|---|---|---|
| 80 | 7 | 0.505 s | 1.084 s |
| 80 | 11 | 0.630 s | 0.556 s |
| 80 | 23 | 0.539 s | 0.546 s |
| 84 | 11 | 0.753 s | 0.754 s |
| 84 | 23 | 0.755 s | 0.808 s |

第一次 80 路的差异是单次波动，三个种子合起来两种策略在 4% 之内，同一策略换种子的波动
（贪心 0.51 到 0.63 s）比策略之间的差异大。

### 结论

单卡 H20、o4.5 F16 权重、这个流量形态下，决策延迟 p95 不超过 1 秒的上限是 **80 路，84 路
是拐点，88 路不达标，两种策略一样**。自适应策略在 12 到 72 路把 p95 降 5% 到 25%，但那个区间
本来就有 3 到 10 倍余量；在决定上限的 80 到 88 路，它测不出差别。

原因在执行器 trace 里（`OMNI_EXEC_TRACE=1`，88 路）：

- 队列几乎不空。贪心策略的 batch 平均 12.7 个 step、p95 46 个，81% 的 batch 里已经带着 prefill。
  没有碎片可以合，预测到达再扣住也就无从省下 forward。自适应策略这时每次只扣 3.5 ms、
  一场只扣 263 次，宽度只比贪心多 2% 到 10%。
- 每次 forward 的成本由 prefill token 数决定，每 token 0.75 到 0.8 ms，与合了多少 step 无关。
  88 路每秒进 88 帧、每帧 11 token 的 prefill，光这一项就是每秒 700 到 800 ms 的 GPU 时间。
  这是算力墙，不是调度墙。
- 把 p95 顶过 1 秒的是个别耗时 1.7 秒的 batch（形状正常，21 到 231 token），不是调度间隙。
  排队等待的 p99 只有 300 ms。

服务时间曲线在 KV 位置 0 测的是乐观值。把 sequence 先填到位置 800 再测（q8_0 KV，4096
上下文）：单步 decode 从 6.8 ms 变 17 ms，32 路 decode 从 17 ms 变 82 ms，32 路 decode 加
32 路 prefill 混在一个 forward 里 502 ms。真实负载下 GPU 时间大部分花在 attention 读 KV 上，
合批省下的固定开销占比比曲线暗示的小得多。自适应策略的固定开销估计（在线从单步单 token
的 batch 学）在高负载下确实收敛到了 26 ms，所以它的判断没有错，只是没有东西可省。

真正把这张卡的上限从 24 路抬到 80 路的是这三件事，都不是调度：pool 模式建会话时不再清掉
所有人的 KV（之前 16 路以上必崩，且被清空上下文的会话每帧都说话，把负载形态整个改了）；
Token2Wav 挪到独立的卡（它每秒语音要 100 到 300 ms GPU，与 LLM 同卡时十几路就把卡吃满）；
每 sequence 上下文从 8192 降到 4096 并用 q8_0 KV，96 槽才放得进显存。

### 断流

各表里的 starve 列没有列出，因为它在这批数据里不是容量信号。修掉播放缓冲的初始状态后，
剩下的断流会话都是同一个形状：每次模型开口的头 0.4 到 0.5 秒。第一个 Token2Wav chunk
在决策后 0.3 到 0.4 秒到达，第二个要再等 1 秒，中间客户端没东西可放。8 路和 88 路一样，
是 Token2Wav 流水线的启动延迟，与并发无关。

## 路上修掉的四个缺陷

这四个都不是 batching 本身的问题，但不修就没有可信的对比。

**合批后取 logits 的下标错了。** executor 把「输出行号」传给 `llama_get_logits_ith`，而它要的是
「batch 内 token 下标」。batch 里只要有一个多 token 的步，后面的 sequence 拿到的就是别人的
logits，日志里每次运行成千条 `invalid logits id`。这个 bug 从 stage 2 引入共享 executor 时就在，
之前所有多 session 结果都是在它之下测的。修完用 `--verify 16` 把 16 路合批（纯 prefill、纯
decode、混合、各 sequence 位置不同）逐一与单步基准比，argmax 全部一致，错误行归零。

**pool 模式建会话时把所有人的 KV 都清了。** slot 复用时对 `ctx_llama` 调 `llama_memory_clear`，
而开了共享 executor 后所有 slot 的 `ctx_llama` 指向同一个 context。每来一个新会话，正在跑的
会话 KV 全部被清空，并且与 worker 正在执行的 `llama_decode` 没有任何同步。后果两条：一是跑到
16 路以上必定在某次 slot 签出后段错误（三份日志同一个签名）；二是被清过 KV 的会话上下文成了
空的，模型从此几乎每帧都说话。前几轮对比里 speak 帧占 60% 到 96%、session 分成「一直说」和
「一直听」两类，就是这个原因，修完后 speak 帧占 1% 到 2%。改成只清本 slot 自己的 sequence，
经 executor 与 worker 串行。

**多字节字符切在 token 边界时 JSON 序列化抛异常。** 服务端整个进程 abort。改用替换式的
错误处理。

**播放缓冲从会话建立就开始计欠账。** loadsim 的缓冲模型初始不在倾听态，一个模型全程只听的
会话被记成 100% 断流。改成初始倾听，并且收到 listen 时结清上一轮的欠账。

另有一个没修的：Token2Wav 多卡轮转（`OMNI_T2W_DEVICES`）在 session 关闭时堆损坏，日志在
`/data/wangkaiqi/logs/crash-t2w-multiowner-20260917.log`。本轮 Token2Wav 单卡。

## 复现

```bash
# 节点上，chroot /proc/1/root 后
cmake --build build-cuda --target llama-omni-server llama-omni-perf-exec-cost -j 32

# 服务时间曲线（2 号卡空闲时）
CUDA_VISIBLE_DEVICES=2 OMNI_EXEC_UBATCH=512 OMNI_EXEC_KV_UNIFIED=1 \
  build-cuda/bin/llama-omni-perf-exec-cost -m <gguf> --widths 1,4,8,16,24,32 --lens 1,5,12,80 --nseq 32
# 合批正确性
build-cuda/bin/llama-omni-perf-exec-cost -m <gguf> --verify 16 --widths "" --nseq 17

# 服务：/tmp/srv.sh <slots> <mixed|adaptive>，布局见脚本头部注释
# 压测：/data/wangkaiqi/loadsim-run/node-sweep2.sh <policy> 12 16 20 24（每个点重启服务）
```
