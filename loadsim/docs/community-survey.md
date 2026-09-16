# 社区调研:实时双工多 session 压测怎么建模

调研时间 2026-09-15。回答一个问题:给 MiniCPM-o 这类实时双工服务做多 session 压测时,
社区用什么到达模型、什么会话时长分布、什么指标,有没有现成工具。

每条都标了来源。**「未核实」的条目是我自己的背景知识,不要当依据用。**

---

## 结论

**没有现成工具能直接拿来用。** 没有一个公开的负载发生器会开一个 WebSocket、保持一个
音频会话、按到达分布投放 N 个这样的会话。所有工具都以「请求」为单位思考,不是以
「会话」为单位。

但有一个研究仓库值得先读:**Metronome**(19PINE-AI),它是唯一公开的、按 session 数
报告实时双工容量的工作,而且它踩过的坑和我们的直接相关。

另外有一个负面结论对我们有利:**没有任何公开的 N-session 扩展曲线**——FDB 全家、
τ-voice 以及我查到的每个评测框架都是单会话的。这块是空白。

---

## 1. Metronome:唯一按 session 数报容量的工作

仓库 `github.com/19PINE-AI/metronome`,论文《Bound the Cache, Keep the Beat for
Real-Time Interaction Model Serving》。链路是真实客户端 → Go gateway(`full_duplex`)
→ gRPC → vLLM/Moshi worker,真实音频、持续多客户端负载。

它给的容量(单张 Blackwell 卡,p99 ≤ 1 秒):

| 模型 | 容量 | 路径 |
|---|---|---|
| MiniCPM-o-4.5 | 48 | windowed-8s(vLLM 0.19,旧的顺序扫描口径) |
| MiniCPM-o-4.5 | ~96 | streaming,append-to-resident-KV,fresh-per-point 复验 |
| Qwen3-Omni-30B-A3B-FP8 | ~64–128 | windowed / streaming(streaming 报 ≥160) |
| Qwen2.5-Omni-7B | ~7 | encoder 受限 |
| Moshi | 16 | 80ms 帧预算;streaming 报 ≥32 |

**四条要直接吸收的东西。**

**一、实时双工的失败模式是「静默的悬崖」。** 论文的核心发现:per-session KV 每帧增长,
直到引擎的 block pool 饱和、调度停滞。三个特征——是内存事件不是算力事件;是亚稳的
(同样的五分钟跑,崩不崩取决于 run-to-run 的池填充速度差异);**而且是静默的:停滞的
引擎仍然按时返回空帧,延迟和 deadline-miss 指标全程显示健康,电话只是安静了**。

这一条直接支持我们工具的核心设计:按「收到多少秒音频 vs 过了多少秒墙钟」算播放缓冲,
而不是相信服务端自报的延迟字段。论文给出的是同一个判据从服务端视角的版本。

**二、帧投递完整性优于服务端自报的延迟。** 同一份材料里记录了 Moshi 自报的 `gpu_ms`
没有同步,导致「0% miss」掩盖了一次 2 倍节奏滑移。

**三、同一模型在不同 KV 策略下容量差 2–6 倍。** MiniCPM-o 在 windowed-8s 与常驻
minute-level context 之间就有这个量级的差异。**报容量必须绑定 KV 策略**,否则数字没有
意义。我们用 demo 默认配置压出来的数,只能配上「这是 demo 默认配置下的数」这句话。

**四、方法纪律:每个 N 用全新进程单独跑。** 他们公开订正过一次错误结论——90 秒扫描
显示 N=128 时突发容量崩溃,后来发现是「顺序扫描 + 外部 GPU 争用」的产物,单点重跑
就平了。不同并发之间顺序复用同一进程状态会得到假结论。

同一份材料里还记录了一个对我们有直接参照价值的对照:panel 1,新鲜 N=96 跑 300 秒,
vanilla vLLM-realtime 的每帧 p50 从 0–30 秒的 2–5 毫秒漂移到 270–300 秒的 1601 毫秒;
而 Metronome 的 in-engine windowed-KV 全程 1–2 毫秒。**「短时高并发没事、几分钟后垮」
这个形态,正是只跑 30 秒的压测永远发现不了的。**

---

## 2. 会话到达过程:泊松是行业默认,闭环/开环才是真正的分界

**已核实。** 推理侧压测工具的分野不是「选哪个分布」,而是「闭环还是开环」:

- **vLLM `benchmark_serving` / `benchmark_serving_multi_turn`**:`--request-rate R`
  走泊松到达,代码路径就是 `poisson_sleep(args.request_rate, ...)`,即均值为 1/R 的
  指数间隔。`--request-rate inf` 表示一次性全发。
- **evalscope perf**(ModelScope):把这对概念作为设计轴明确写出来。闭环用信号量限住在途
  请求数(`--parallel`),有背压,服务时间相关;开环用 `--open-loop` 按 `--rate` 的
  泊松过程发,无背压。它还提供 `--sla-auto-tune`,二分搜索满足延迟约束(如 `avg_ttft`)
  的最大速率或并发。这是查到的对闭环/开环讲得最清楚的一份。
- **vLLM 的 goodput/SLO 机制**:请求满足 TTFT/TPOT/E2EL 阈值即标为 good,goodput 是
  每秒 good 请求数。

**未核实(背景知识,别当依据):** MLPerf Inference LoadGen 的 Server 场景是经典的开环
泊松发生器(目标 QPS,泊松查询到达)。Locust 和 k6 默认闭环(VU 等响应),
k6 的 `ramping-arrival-rate` / `constant-arrival-rate` 执行器是开环出口,Locust 的是
`LoadTestShape`。Vegeta 是恒定速率开环;wrk2 是恒定吞吐并做 coordinated-omission 修正,
但仍属闭环。GenAI-Perf 默认并发驱动。llmperf 是 `--num_concurrent_requests` 闭环。

**coordinated omission** 是所有闭环音频压测工具都会有的失真,值得点名:客户端等响应
所以不会在服务端慢的时候继续发压,于是慢的那段时间被「跳过」了,压出来的延迟好看得多。
wrk2 和 LoadGen 处理了这个问题,Locust 和 k6 的默认 VU 模式没有。

---

## 3. 会话时长分布:没有可引用的拟合

**已核实,而且能查到的很少。** 没有任何语音服务论文公布过通话时长的分布拟合。有的:

- 一家厂商自己的 AI 前台电话遥测:中位时长 **49 秒**,p90 **143 秒**,最长的十分位
  占了**全部通话时间的 32.9%**(147,885 通,2026 年 8 月 10 日到 9 月 6 日)。实际是重尾,
  但该页面没有拟合任何参数形式。
- 对比性的处理时长:同类事务查询下,AI 语音 agent 30–90 秒,人工坐席 3–8 分钟。
- 查到的最有用的「设计数字」不是分布而是上限:τ-voice 把 `max_steps_seconds` 设为
  **1200 秒(20 分钟)**作为模拟对话的最大长度。

**背景知识:** 话务工程里电话保持时长传统上拟合对数正态,而 49 秒 / 143 秒这一对看起来
就是「对数正态主体 + 重尾」。**但没有语音领域的论文断言过这个拟合,所以任何对数正态的
参数选择都是我们自己的建模决定,不是有出处的东西。**

**处置:** 工具默认保持对数正态,但报告里要标注这是假设。要定准,得用自己服务上会话时长
的直方图去拟合。这是本次调研的已知空白之一。

---

## 4. 会话内的说话节奏:这部分有几十年历史、有实测数字

这是全篇唯一有扎实定量文献的部分,而且全部来自前 LLM 时代的电话话务工程。

**已核实:**

- **Lee & Un, "A Study of On-Off Characteristics of Conversational Speech," IEEE Trans.
  Comm. 34(6), 1986。** 50 分钟电话语音,在 200ms hangover 或 fill-in 下测得:
  **平均说话段 1.11–1.30 秒,平均静音 2.22–2.41 秒,说话段频次 16.67 次/分,
  语音活动比 31.5–37.8%。** hangover > 200ms 时静音拟合指数分布、说话段拟合
  「常数 + 指数」;fill-in > 200ms 时两者角色互换。hangover 增到 300ms 时说话段频次
  下降约 5 倍(72 次/分 → 14 次/分),语音活动比升到 27% → 40%。作者明确指出静音的
  拟合比说话段差。
- **Brady 模型**(引自一份 Purdue 学位论文):说话段长度取**对数正态,均值 4.14 秒、
  众数 0.25 秒**(μ=0.485, σ=1.871);**响应时间正态,均值 0.41 秒,标准差 0.584 秒**。
  对抢话最有用的一点:响应时间为负被定义为「第二个说话人在第一个还在说的时候就开口」,
  也就是打断。这是一个可以直接用的抢话模型。同一份材料里也记了一个矛盾:Norwine &
  Murphy(1938)测得平均说话段 4.15 秒,而 Brady 测到 1.34 秒,差别来自他们对「说话段」
  的定义会跨过 250ms 以下的停顿。
- **Wlodarczak & Wagner, INTERSPEECH 2013**:实测的间隔与重叠分布对分割用的静音阈值
  **极其敏感**。把 150ms 以上的静音桥接掉会虚增同一说话人内部的重叠计数,并让
  重叠者/被重叠者的角色互换。相关:Heldner 2011 把感知上的间隔/重叠检测阈值定在
  约 120ms。
- **Bosch, Oostdijk & de Ruiter 2004**:面对面交谈中的停顿比电话交谈中长最多 4 倍,
  条件在功能上可比,说话人配对相关系数 ≥0.88。**如果我们压的是设备端/环境语音 agent
  而不是电话,这一条适用。**
- **聚合状态的数学模型**:对 n 个说话人,同时活跃的数量是一个**生灭(Markov)过程**,
  1/λ = 平均静音,1/μ = 平均说话段;而且**当 n ≥ 25 时该过程趋近泊松**(Weinstein)。
  这是把单会话说话节奏接到服务端负载的那一环。

**对我们工具的直接含义:** 我们 `SpeechPattern` 现在的默认是说话段 1–5 秒(均值 3 秒)、
静音 0.5–3 秒(均值 1.75 秒),算出来的语音活动比约 **0.63**,而 Lee & Un 实测是
**0.32–0.38**。**我们的默认设置让用户说话的时间差不多是真实情况的两倍**,上行负载因此
偏高。默认值应当改到接近实测:说话段均值约 1.2 秒、静音均值约 2.3 秒,语音活动比
落在 0.35 附近。

---

## 5. 全双工合成客户端:可借鉴的两个实现

**已核实。** τ-voice 和 Full-Duplex-Bench v2 各自解决了一半问题。

**τ-voice**(sierra-research/tau2-bench)是最好的起点。基于 tick 的模拟器:
`tick_duration_seconds = 0.20`,离散时间步驱动连续 PCM/μ-law 流,`VoiceStreamingUserSimulator`
用 ElevenLabs TTS 合成用户音频,并且有具名的轮次阈值:

| 参数 | 默认值 |
|---|---|
| tick_duration_seconds | 0.20 秒 |
| wait_to_respond_threshold | 1.0 秒 |
| backchannel_min_threshold | 3.0 秒 |
| max_steps_seconds | 1200 秒 |
| telephony_rate | 8000 Hz μ-law |

它对 OpenAI Realtime、Gemini Live、xAI Grok Voice、Amazon Nova Sonic、Qwen Omni 说
各自的原生 WebSocket 音频协议,也有 LiveKit 级联的 STT→LLM→TTS 路径,还提供
`regular`(真实/带噪)与 `control`(干净)两档语音复杂度预设。**它是为单 agent 评测
造的,不是负载发生器,但它是现存的、最接近「合成全双工客户端」的东西。**

**Full-Duplex-Bench** 系列(github.com/DanielLin94144/Full-Duplex-Bench)把重叠场景的
分类法编码了进去:

- **v1.0**(arXiv 2503.04721):pause handling、backchanneling、turn-taking、interruption
  四个维度。指标包括 **Takeover Rate (TOR)**、附和频次、以及**模型附和时机与人类时机
  之间的 Jensen–Shannon 散度**;一段音频若短于 1 秒且少于 2 个词才算附和。
- **v1.5**(arXiv 2507.23159):四种脚本化重叠场景——Interruption(200 条合成样本)、
  Backchannel(99)、Talking to Others(100,模拟离轴:-8dB、4kHz 以上高架衰减 -5dB、
  45ms/-6dB 与 120ms/-12dB 两处反射)、Background Speech(100,-15dB、3kHz 低通、
  100ms 回声 -10dB)。增加了 stop latency、response latency 和韵律指标。
- **v2**(arXiv 2510.07838):由 WebRTC 编排器强制的**统一线格式:48 kHz、16 位、
  单声道 PCM、严格 10ms 帧(960 字节)**,双向连续,并要求 adapter **在缓冲欠载时
  补静音**。两种节奏模式由 `turn_detection` 驱动:Slow =
  `{server_vad, create_response: true, interrupt_response: false}`,Fast =
  `interrupt_response: true`。由 GPT-Realtime 扮演一个脚本化的 Examiner,分步推进目标
  并主动打断。
- **v3**(arXiv 2604.04847):真实非流利语音下的工具调用,据称 100 例 / 79 场景 /
  12 说话人 / 4 领域,延迟单独报告。

**两个该抄的具体机制:10ms 统一帧,以及 `interrupt_response` 标志位。**

**关于帧长:** 我们自己上行用的是 80ms 帧。OpenAI Realtime 的默认 server VAD 正好是
每 80ms 一个音频 chunk(采样率 24kHz、960 采样/块),所以 80ms 是有依据的取值,
不是随手定的。τ-voice / FDB v2 用 200ms 一个 tick 是评测编排的粒度,不是音频帧粒度。

---

## 6. 指标与 SLO:没有标准,但有可引用的生产预算

**已核实。** 最接近公开数字的是 CallSphere 的生产总结(约每月 28 万会话):

| 指标 | p50 | p95 | 硬上限 |
|---|---|---|---|
| time_to_first_audio | 480 ms | 780 ms | 1200 ms |
| time_to_tool_call | 240 ms | 410 ms | 800 ms |
| time_to_response_complete | 1.6 s | 2.9 s | 4.5 s |

他们的抢话检测是**联合 VAD 重叠检测**:任何窗口内用户与 agent 音频同时超过 VAD 阈值
持续 >150ms 就算一次重叠,可接受的重叠率是**每次通话 <2%**。他们还要求
`turnDetection.silenceDurationMs` 落在 250–400ms 区间(他们跑 320ms),并把
`audio_interrupted` → `response.cancel` 当作强制行为。**这是厂商博客不是同行评议,
但它是查到的唯一具体的语音 agent SLO。**

**Full-Duplex-Bench 的标准化指标**是对话行为的分类标签(RESPOND / RESUME /
UNCERTAIN)、stop latency、response latency,以及附和位置的 JSD-to-human。
**它全文没有 underrun 或 glitch rate 这类指标。**

**OpenAI 自己的 Realtime 翻译指南**点了该测的运行项:首段翻译音频延迟、结束话音延迟、
字幕时序、重连行为,并且明确要求「持续发送音频,包括短语之间的静音」。

**未核实(背景知识):** ITU-T G.114 的单向延迟分档(≤150ms 理想,≤400ms 可接受)。
**没有查到语音 agent 专用的标准,也没有查到把 inter-chunk gap / underrun / glitch rate
作为具名指标的工作——这看起来是一块需要我们去填的空白。**

---

## 7. 现成工具:逐个判定为什么不能直接用

| 工具 | 做什么 | 为什么不能直接用 |
|---|---|---|
| Metronome 的评测 harness | 实时双工容量评测,真实客户端 + Go gateway + gRPC | 绑死作者自己的服务栈,不是通用发生器 |
| τ-voice(tau2-bench) | 实时语音对话评测,tick 200ms,六个厂商的原生音频客户端 | 单会话;缺多会话编排、到达过程、按会话的负载记账 |
| Full-Duplex-Bench v2 | WebRTC 编排,统一线格式,adapter 模式,10ms 节奏 | 单 Examiner 对单 Evaluatee,没有 N 会话或负载的概念 |
| vLLM `benchmark_serving_multi_turn` | 有状态多轮客户端,含 `poisson_sleep` | 纯文本 |
| evalscope perf | 泊松开环 + `--sla-auto-tune` 二分搜索 | 协议是文本 LLM 的 HTTP |
| k6 / Locust / Vegeta / GenAI-Perf | 通用 HTTP 负载 | 需要自己包一层 Realtime 协议会话脚本(未核实具体可行性) |

**缺口就是:没有人发布过 Realtime API 的会话到达负载发生器,也没有工具把「会话被一直
保持着、音频持续流动」这件事记进账里。**

---

## 8. 对我们工具意味着什么

按优先度排:

1. **改 `SpeechPattern` 的默认值。** 现在的语音活动比约 0.63,实测是 0.32–0.38,
   我们的默认让上行负载差不多翻倍。改成说话段均值约 1.2 秒、静音均值约 2.3 秒。
   这是本次调研里唯一一个能立刻修、且有实测依据的项。
2. **到达模型保持泊松为默认**,与全行业一致,不需要辩护。同时把闭环/开环的区分
   写进文档——我们的 `--reuse` 是闭环口径(维持并发),不补位是开环口径(维持到达率),
   两者测的不是一回事。将来可以把 `--burstiness` 做成 gamma 形状参数的连续量。
3. **把「不同并发之间必须用全新进程单独跑」写进运行纪律。** Metronome 为这件事公开
   订正过一次结论。
4. **播放缓冲水位作为主指标是对的,而且是块空白。** Metronome 从服务端侧用「帧投递
   完整性」、我们从客户端侧用「缓冲水位」,都优于相信服务端自报的延迟。查到的工作里
   没人把 underrun 作为具名指标。
5. **跑够时长。** Metronome 的漂移数据(30 秒时 2–5 毫秒、300 秒时 1601 毫秒)说明
   30 秒的压测看不见真正的失败模式。默认时长应当拉到几分钟。
6. **会话时长分布保持对数正态但标注为假设。** 可参照的遥测是中位 49 秒 / p90 143 秒 /
   最长的十分位占全部通话时间 32.9%。
7. **顺手做的对照:** 同一份负载分别对 gateway 和直连 worker 各跑一次,区分「排队
   饱和」和「模型算不动」。
8. **报容量必须绑定 KV 策略。** 同一模型在 windowed 与常驻 context 之间差 2–6 倍。
