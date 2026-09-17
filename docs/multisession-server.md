# 多 session pool 服务端实现与端到端验证

2026-09-17。分支 `feature/server-multisession`，基线 `exp/o5-multisession-capacity`。

## 做了什么

`exp/o5-multisession-capacity` 把引擎层的多 session 能力做完了:一次 `omni_init`
加载全部权重,再用 `omni_session_clone` 复制出 N-1 个会话上下文共享这些权重,每个
clone 只持有自己的 KV cache 与工作线程。但 `server-omni.cpp` 一行没改——单个全局
`omni_context`、`n_parallel` 硬编码 1、`/backend` 的 WebSocket 共享同一个上下文。
引擎能跑并发,HTTP 层把并发挡住了。

这个分支补齐服务层,改动五个文件。

### 1. 会话池(session.h / session.cpp)

`SessionPool` 持有 N 个 `omni_context*`(首个是 root,拥有全部权重;其余是 clone)。
`checkout_slot()` 阻塞到有空闲槽位,`return_slot()` 归还并唤醒等待方。
`shutdown()` 里逆序释放:clone 先、root 最后,因为权重归 root。

`SessionManager` 增加 `pool_sessions_`(会话 id 到会话的映射)与 `session_slot_`
(会话 id 到槽位号的映射),与原有单会话路径并存。`-np 1` 时走原路径,行为不变,
已有部署不受影响。

### 2. WebSocket 处理(ws_handler.cpp / ws_handler.h)

`handle_ws_backend` 增加 `SessionPool * pool` 参数。非空时:

1. `checkout_slot()` 取一个空闲槽位,若池已关闭则直接拒绝连接
2. 发 `session.queue_done`
3. 读 `session.init`
4. `reset_octx_for_session` 清掉上一个会话的状态(线程、KV cache、计数器)
5. `omni_duplex_session_begin` 初始化系统提示词与语音克隆
6. 循环:收 `input.append` → `omni_duplex_push_frame` → `omni_duplex_wait_next_frame`
   → 按 `is_speak` 发 listen 或 text delta,音频由 `audio_output_cb` 异步发出
7. 断开时 `omni_duplex_session_end` + `omni_prepare_for_reuse`,再归还槽位

`pool == nullptr` 时整段跳过,走原来的单会话代码。

### 3. 启动与健康检查(server-omni.cpp)

`-np N` 是并发会话数。`build_session_pool()` 在启动时建池:先 `omni_init` 加载
权重,再 `omni_session_clone` 复制 N-1 份,最后 `omni_enable_llm_exec` 开启批量
decode(把所有槽位的解码合成一个 batch)。任一步失败都不会让服务起不来,而是退回
单会话模式并在日志里说明。

`/health` 增加 `sessions` 字段,报告池的模式与容量:

```json
{"engine":"comni","sessions":{"capacity":2,"mode":"pool"},"status":"ok"}
```

压测工具读这个数就能按服务端实际容量定并发,不用猜。

### 4. `session.queue_done`(protocol.h / protocol.cpp)

原实现把第一条收到的消息当作 `session.init`,不发任何排队事件。这会让任何按
文档协议写的客户端死锁:客户端等 `queue_done` 才发 init,服务端等 init 才回话,
双方都卡在读上,压测端报「排队超时、零上行帧」,而服务端日志里一句错都没有。

加了 `make_session_queue_done()`,在第一次读之前发出去。排队事件必须在读之前,
这是由协议顺序决定的。

### 5. `/sessions/:id/close` 的池路径

池模式下每个槽位有独立的上下文,关闭时不需要持有全局互斥量。停推理、排空线程、
归还槽位三步走。单会话路径保持原样(仍持锁保护那唯一的共享上下文)。

## 端到端验证

节点:南京 H20 节点(`vm-7-13-ubuntu`,8×H20 96 GB),单卡。
权重:MiniCPM-o-4.5 F16(`/data/omni-convert/gguf/`)。
服务:`llama-omni-server -np 2 -c 8192`,启用 TLS。

### 单会话

| 指标 | 数值 |
|---|---|
| 建会话延迟 | 2.33 s |
| 排队等待 | 0.17 s |
| 首帧音频延迟 | 0.61 s |
| 上行帧数 | 564 |
| 下行音频 delta | 279 |
| 断流占比 | 0% |
| 语音活动比 | 0.399 |

### 两路并行

| 指标 | 数值 |
|---|---|
| 峰值并发 | 2 |
| 排队超时 | 0 |
| 建会话延迟 中位/95分位 | 2.47 / 2.54 s |
| 首帧音频延迟 中位/95分位 | 0.62 / 0.67 s |
| 断流占比 | 0% |
| 语音活动比 | 0.392 |

服务端日志是并发确实发生的证据:

```
11.15.754.012 I WS /backend (pool): checked out slot 0
11.15.754.686 I WS /backend (pool): checked out slot 1
12.07.004.297 I WS /backend (pool): session ... slot 0 disconnecting
12.07.107.026 I WS /backend (pool): session ... slot 1 disconnecting
```

两次 checkout 相隔 0.7 毫秒,两个会话并行运行约 52 秒后各自归还槽位。这排除了
「两次串行复用同一个槽位」这种看着像并发其实不是的情形。

## 排障记录:两个把「服务没起来」伪装成别的问题的坑

**一、TLS 证书没给,服务静默退出。** 这个构建启用了 OpenSSL,`httplib::SSLServer`
在构造时加载证书;证书路径为空则 `is_valid()` 返回假,`listen()` 立刻返回 0。
外层看到的是「进程正常退出、退出码 0、没有任何错误日志」,很容易误判成崩溃或
内存不足。判据:`svr.listen` 的返回值必须检查并记日志,返回 0 就是没监听成功。
HTTP 模式或提供有效证书都可解决。

**二、池子初始化被写了两遍。** `server-omni.cpp` 里有两段几乎相同的 `build_session_pool`
调用,第二次触发 `SessionPool::init_slots called twice` 抛异常。这类重复在编辑
过程中很容易留下,而症状(启动即崩、堆栈指向池子)看起来像池子本身的缺陷。
第二段已删除。

## 还没做的

- **只有 `full_duplex` 走池。** 池分支没处理 `turn_based`,那个模式仍走单会话路径。
- **没有队列深度上限。** `checkout_slot()` 无限等待。真实网关需要有界队列,满了
  要回 `session.queued` 带排队位置,而不是让客户端干等。
- **没有在池模式下验证过载。** 本次只测到服务端容量(2 路),没测超出容量时的行为。
- **权重是 F16 不是量化版。** 96 GB 卡放得下,但真实部署会用 Q8_0 或 Q4_K_M,
  每会话增量显存也会不同。

## 复现

```bash
# 节点上编译(用宿主机自己的工具链,chroot 进 /proc/1/root)
cmake -B build-cuda -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build-cuda --target llama-omni-server -j 32

# 起服务(证书必需,否则 listen 直接返回 0)
./build-cuda/bin/llama-omni-server \
  -m /data/wangkaiqi/o45gguf/MiniCPM-o-4_5-F16.gguf \
  -np 2 --port 8018 -c 8192 \
  --ssl-cert-file certs/cert.pem --ssl-key-file certs/key.pem

# 压测(loadsim 打 /backend 而不是 /v1/realtime)
python -m loadsim.cli run \
  --url "wss://127.0.0.1:18018/backend" --insecure \
  --arrival all_at_once --concurrency 2 --duration 100 \
  --lifetime-mean 80 --lifetime-dist fixed --max-wait 150 \
  --user-audio user_audio.wav --out /tmp/e2e.json
```
