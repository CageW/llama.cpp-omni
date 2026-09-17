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

## 端到端对比

服务端 24 个 slot，LLM 与 TTS 各一个共享 executor 在 2 号卡，编码器在 3 号卡，Token2Wav 独占
4 号卡。loadsim 在节点内直连，每帧 1000 ms（与产品前端一致），trickle 到达 15 秒内铺开，每
session 固定 70 秒，整场 95 秒，随机种子固定，system prompt 与 `length_penalty` 用产品默认。
判据：决策延迟（从发出第 k 帧到收到模型对第 k 帧的 listen 或 text）p95 不超过 1 秒。

RESULTS_PLACEHOLDER

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
