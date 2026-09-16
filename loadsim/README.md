# loadsim

多 session 实时双工压测工具,用于测量 MiniCPM-o 这类实时语音模型在并发场景下的表现。

## 核心问题

实时语音和普通请求的根本区别在于:输出是一段连续的音频流,客户端要边收边播。服务端算得慢一点、音频晚到一点,只要缓冲区里还有货就不会被听出来;一旦缓冲区见底,声音就断了。

所以主指标不是平均延迟,而是**播放缓冲见底的时长占比**(starvation ratio),以及延迟的分位数尾巴。

## 快速开始

```bash
# 本机自检(不需要 GPU,验证工具本身)
python -m loadsim.cli selftest

# 对真实服务发压(8 路并发,泊松到达,跑 5 分钟)
python -m loadsim.cli run \
  --url ws://127.0.0.1:8006/v1/realtime?mode=audio \
  --arrival poisson --concurrency 8 --duration 300 \
  --user-audio tests/cases/common/user_audio/000_user_audio0.wav \
  --out results/poisson-c8.json
```

## 到达模型

支持五种会话到达模式:

- **poisson**: 泊松到达(指数间隔),排队论标准口径,真实流量的理想近似
- **ramp**: 线性爬升,闭环工具的常见模式
- **burst**: 突发尖峰,复现瞬时高并发场景
- **trickle**: 慢速涓流,用于容量摸底
- **all_at_once**: 一次性投放,压力上限测试

`--reuse` 开启补位模式:会话退出后自动补位,把并发顶在目标值。不开时只投一轮,稳态并发会低于目标值(见 memory 记录)。

## 说话节奏

压测客户端按真实电话语音的实测统计生成说话节奏:

- 说话段(talkspurt):0.6~1.8s,均匀分布
- 静音段(pause):1.2~2.6s,均匀分布
- 语音活动比(speech activity):约 32~38%,与电话语音实测一致

默认静音段照常上行静音帧(真实浏览器/原生客户端的行为)。`--no-silence-frames` 切换到移动端省流量模式(静音段完全不上行)。

## 输出指标

报告包含三个层次:

### 1. 会话层(每个 session 的一生)

- `setup_latency_s`: 从发起连接到拿到 session.created,包含排队等待
- `queue_wait_s`: 排队等 worker 的时长
- `ttfa_s`: time to first audio,从会话建好到收到第一帧音频
- `session_duration_s`: 会话总时长
- `frames_sent / audio_sent_s`: 上行音频量
- `audio_deltas / audio_received_s`: 下行音频量
- `listen_events`: 模型进入倾听状态的次数
- `underrun_total_s / underrun_events`: 播放缓冲见底的总时长与次数
- `starve_ratio`: 断流时长占会话时长的比例

### 2. 汇总指标

- `concurrency_peak / concurrency_steady_mean`: 峰值并发与稳态并发均值
- `starving_session_ratio`: 断流会话占比
- `starve_ratio_mean / p50 / p95`: 断流时长占比的均值、中位数、95 分位
- `setup_latency_p50_s / p95_s`: 建会话延迟分位数
- `ttfa_p50_s / p95_s`: 首帧音频延迟分位数

### 3. 并发曲线

`concurrency_series` 记录每秒采样的实际并发数,用于验证到达模型是否达到预期。

## 播放缓冲模型

`PlaybackBuffer` 是一个库存模型:

- 收到一秒音频就多一秒库存
- 过了一秒墙钟就烧掉一秒库存
- 库存为负说明没货可播,负到多少就是已经断流多久

**关键设计**:区分两种状态:

1. **speaking**(有 audio delta 下发):tick 消耗库存,见底算断流
2. **listening**(收到 `kind=listen`):暂停 tick,不消耗库存,等下一轮 audio delta

这是双工协议的核心语义:`kind=listen` 表示「模型进入倾听状态,客户端应当停止播放」。不处理 listen 会把协议正常的倾听状态误判成断流。

## 项目结构

```
loadsim/
├── cli.py              # 命令行入口
├── runner.py           # 压测编排,投放与收尾
├── arrival.py          # 到达模型,生成投放计划
├── client.py           # 单个会话客户端
├── activity.py         # 说话节奏生成
├── metrics.py          # 会话指标与播放缓冲
├── fake_server.py      # 假服务端,用于自检
└── tests/
    └── test_arrival_and_metrics.py  # 单元测试
```

## 验证数据

| 场景 | listen 事件 | audio delta | 断流占比 | 说明 |
|------|-------------|-------------|----------|------|
| 真实单会话 | 371 | 6 | 0.0% | 正确:倾听时暂停消耗 |
| 真实 4 路并发 | 1135 | 14 | 33.3% | 6/9 会话无断流,3/9 真断流 |
| 假服务端 selftest | 0 | 887 | 66.7% | 预期断流:并发高时故意变慢 |

假服务端不发 listen 事件(用来验证到达模型和并发采样,不模拟双工状态切换)。

## 依赖

- Python 3.10+
- websockets
- numpy

开发依赖:
- pytest
- wave(标准库)

## 相关文档

- `CHANGELOG_listen_fix.md`: listen 事件处理修复详情
- Memory: `omni-duplex-load-sim-arrival-model` - 到达模型设计
- Memory: `omni-duplex-load-listen-fix` - listen 状态误判修复
