# H20 上 MiniCPM-o5 LLM 代理压测（2026-09-03）

目的：在实现多 session 之前，先用与 MiniCPM-o5 backbone 同形状的公开模型
（Qwen3.5-35B-A3B，Q8_0，40 层 = 10 full-attention + 30 gated-delta-net，hidden 2048，
2 KV head × 256，256 专家选 8）在 H20 上测出两件事：
每个 sequence 占多少显存；prefill/decode 随 batch 的吞吐曲线。

工具：上游 llama.cpp `llama-batched-bench`（sm_90 编译，flash attention 开）。
原始日志在南京节点 `/data/wangkaiqi/o5cap/e1 e2 e3`，脚本见本目录 `e1.sh e2.sh e3.sh`。

## 显存（单卡，n_seq_max = 256，per-seq ctx 8192）

| 项 | 大小 | 备注 |
|---|---|---|
| 权重 Q8_0 | 34.7 GB | |
| full-attention KV，F16 | 40.0 GB | 每 seq 每 token 20 KiB |
| full-attention KV，q8_0 | 21.3 GB | |
| gated-delta-net 递归状态 | 15.7 GB | 每 seq 62.8 MiB，F32，与 ctx 无关 |
| compute buffer | 1.1 GB | 与 seq 数无关；ub=128 只省 30 MiB 且 prefill 吞吐减半 |

递归状态是 per-session 的最大单项，比 8K 的 q8 KV（21.3/256 = 85 MiB）只小一点。
32K ctx 时单卡只能放 128 seq（KV 43.5 GB + 递归 8 GB）。

## 吞吐（单卡，pp = 100 模拟 omni 每秒 prefill，tg = 16 模拟 speak）

| batch | prefill 100 tok | prefill tok/s | decode ms/step | decode tok/s |
|---|---|---|---|---|
| 1 | 58 ms | 1710 | 5.8 | 172 |
| 8 | 213 ms | 3760 | 13.4 | 599 |
| 32 | 867 ms | 3690 | 32 | 995 |
| 128 | 3.69 s | 3470 | 95 | 1344 |
| 256 | 7.33 s | 3490 | 150 | 1706 |

prefill 在 batch ≥ 4 后平在 3500~3800 tok/s，是单卡 MoE 的算力墙。
decode 到 256 仍在涨，没饱和。4000 token 长 prompt 也是 3800 tok/s，attention 不是瓶颈。

layer split：4 卡 prefill 4400~4800 tok/s（+25%），2 卡 3900~4500（+12%），decode 不变。

## 对 session 上限的含义

duplex 每个 session 每秒必须完成一次 prefill。omni 模式每秒约 100 token，
单卡 3600 tok/s 只够 30~36 个 session；8 卡各一份副本约 250~280 个。
LLM 切到 4 卡两副本反而更少（2 × 4600 / 100 ≈ 92）。
纯语音每秒 17 token，算力允许约 200 个/卡，此时显存（256 seq）先到顶。
