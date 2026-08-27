# TTS Voice Cloning 分阶段设计

## 目标

`llama.cpp-omni` 需要支持从 reference WAV 在线准备 TTS voice cache，并在同一个 session 中使用该 voice 生成语音。阶段二已完成，当前 HTTP path-based 流程已经使用 native C++ frontend，运行时不再依赖 Python frontend 计算 PromptBundle。阶段三补齐了 WebSocket base64 `tts_ref_audio` 的 nanjing H20 E2E，并完成了 Python/C++ TTS 的同 token 输出对比。

当前主流程是：

```text
reference WAV
  -> C++ Token2WavFrontend
  -> PromptBundle
  -> Token2Mel::setup_cache()
  -> C++ Token2Wav streaming
  -> 24 kHz mono PCM16 WAV
```

`Token2WavFrontend` 使用 C++ 完成 WAV 解码、16 kHz resample、StepAudio2 log-Mel、TTS prompt Mel 和 CampPlus fbank；speech tokenizer 和 speaker embedding 使用挂载的 ONNX 权重及 ONNX Runtime 推理。C++ 侧把这三类结果组装成 `PromptBundle`，再交给已经加载的 Token2Mel 和 vocoder model。

## 当前边界

`llama.cpp-omni` 有两条独立的 reference audio 语义：

1. `ref_audio` 进入 APM，用于 LLM system prompt 的 audio embedding。
2. `tts_ref_audio` 进入 Token2Wav frontend，用于生成当前 session 的 voice cache。

HTTP `omni_init` 中未提供 `tts_ref_audio` 时，服务端使用 `voice_audio`。两个字段都没有时，C++ Token2Wav 继续加载模型目录中的静态 `prompt_cache.gguf`。

WebSocket `session.init` 支持 `ref_audio` 和 `tts_ref_audio` 的 base64 float32 PCM。解析逻辑会在只提供 `ref_audio` 时复用该 audio 作为 TTS reference。阶段三已经在 nanjing H20 上验证了 `tts_ref_audio` 的 base64 客户端链路；`full-duplex` 的持续输入和播放时序仍未纳入本阶段验收。

## 阶段二实现

### Native frontend

`tools/omni/token2wav/token2wav-frontend.cpp` 和对应 header 提供以下能力：

- WAV 文件读取和 mono 化；
- 16 kHz resample；
- StepAudio2 speech tokenizer 的 25 Hz audio token；
- Token2Wav prompt Mel；
- CampPlus fbank 和 192 维 speaker embedding；
- shape、dtype、channel 和时间长度校验；
- 从 reference WAV 直接生成 `FrontendPromptBundle`。

speech tokenizer 和 CampPlus 权重通过环境变量配置：

```text
OMNI_T2W_SPEECH_TOKENIZER_ONNX=/path/to/speech_tokenizer_v2_25hz.onnx
OMNI_T2W_CAMPPLUS_ONNX=/path/to/campplus.onnx
```

这两个变量都存在时，`omni_set_tts_reference_audio()` 调用 `Token2WavSession::set_prompt_wav()`。native frontend 失败会让 session 初始化失败，不会静默回退到旧 voice。

### PromptBundle 和 session

`Token2WavSession` 提供：

```cpp
bool set_prompt_bundle(const std::string & prompt_bundle_dir,
                       int n_timesteps = 10,
                       float temperature = 1.0f);

bool set_prompt_wav(const std::string & ref_wav_path,
                    const std::string & speech_tokenizer_onnx,
                    const std::string & campplus_onnx,
                    int frontend_threads = 1,
                    int n_timesteps = 10,
                    float temperature = 1.0f);
```

`set_prompt_wav()` 先运行 native frontend，再调用 `Token2Mel::setup_cache()`。新的 cache 安装前会清空当前 TTS stream 的 `pending_` 和 `wave_tmp_`；Token2Mel 和 vocoder 权重不会重新加载。`reset_to_prompt_cache_gguf()` 保留静态默认 voice 的恢复能力。

PromptBundle 的文件目录格式仍支持 Python adapter 和 C++ 测试：

```text
manifest.json
prompt_tokens_i32.bin
prompt_mel_btc_f32.bin
spk_f32.bin
```

`tools/omni/pyt2w/token2wav_service.py` 保留 `prepare_prompt_bundle`，用于兼容 sidecar 和离线检查；当前 nanjing HTTP server 由 native frontend 直接计算，不经过该 Python bundle 生成路径。

### Server 调用关系

HTTP `omni_init` 收到 `tts_ref_audio` 后执行：

```text
HTTP request
  -> omni_set_tts_reference_audio()
  -> Token2WavSession::set_prompt_wav()
  -> native frontend
  -> Token2Mel::setup_cache()
  -> start C++ TTS worker
```

reference WAV 只在初始化阶段使用。当前实现没有运行中 voice switch API，也没有全局 cache registry。这样可以避免 TTS worker 使用 mutable stream state 时发生 cache 替换。

## API 合约

### HTTP

```json
{
  "media_type": 1,
  "use_tts": true,
  "duplex_mode": false,
  "tts_gpu_layers": 100,
  "token2wav_device": "gpu:0",
  "output_dir": "/path/to/output",
  "voice_audio": "/path/to/llm-reference.wav",
  "tts_ref_audio": "/path/to/tts-reference.wav"
}
```

请求顺序是：

```text
GET  /health
POST /v1/stream/omni_init
POST /v1/stream/prefill       { cnt: 1, audio_path_prefix: ... }
POST /v1/stream/decode        { stream: true }
```

`omni_init` 内部完成 `cnt=0` 的 system prompt prefill，客户端从 `cnt=1` 开始。decode 的 SSE 以一个 `data: [DONE]` 结束，TTS WAV 写入 `output_dir/round_XXX/tts_wav/`。

### WebSocket

WebSocket `session.init` 的字段如下：

```json
{
  "voice": {
    "ref_audio": "<base64 float32 PCM>",
    "tts_ref_audio": "<base64 float32 PCM>"
  }
}
```

`ref_audio` 只影响 LLM APM prompt，`tts_ref_audio` 只影响 Token2Wav。只提供 `ref_audio` 时两条路径复用同一个 audio。frontend 或 cache setup 失败时 session 不激活，并返回 `tts_voice_prepare_failed` 对应的 fail-fast 结果。

## 错误处理

native frontend 必须拒绝以下情况：

- reference WAV 不存在、不可读或没有 samples；
- WAV 无法转换为 mono 16 kHz；
- speech tokenizer 输出为空；
- prompt Mel 不是 80 channels；
- speaker embedding 不是 192 维；
- `T_prompt_mel != (T_prompt_token - 3) * 2`；
- ONNX 权重不可加载或推理输出 shape 不匹配。

C++ session 必须拒绝以下情况：

- PromptBundle manifest 与二进制文件不一致；
- bundle 中存在非 `B=1` 数据；
- `Token2Mel::setup_cache()` 失败；
- `start_stream_with_prompt()` 失败；
- 已有未完成 TTS 请求时尝试切换 voice。

日志记录 stage、session 或输入路径和失败原因，但不记录 audio 内容和 base64。HTTP session 初始化失败时释放新 context，旧 session 不会继续持有半初始化的 voice state。

## 测试

### Python adapter

`tools/omni/pyt2w/test_prompt_bundle.py` 覆盖：

- PromptBundle 的 shape、dtype、layout 和 manifest；
- 16 kHz reference audio 的 loader 兼容性；
- prompt token lookahead；
- 缺少文件和 shape mismatch；
- 整数 Mel 或 speaker embedding 的拒绝。

从仓库根目录运行：

```bash
python3 -m unittest tools/omni/pyt2w/test_prompt_bundle.py
git diff --check
```

### C++ native frontend 和 session

`tools/omni/test/test-token2wav-frontend.cpp` 覆盖 resample、log-Mel、prompt Mel、CampPlus fbank、bundle 组装和真实 reference WAV；`tools/omni/test/test-token2wav-session.cpp` 覆盖两个 voice 的 cache 安装、28-token window 和 stream reset。

### Nanjing H20 HTTP E2E

验证日期：2026-08-26。

| 项目 | 实际值 |
|---|---|
| SSH jump host | `uds-dev1` |
| Kubernetes cluster | `nanjing-h20-rke2.config` |
| Pod | `default/codex-tts-dev-20260826` |
| Node | `vm-7-13-ubuntu` |
| GPU | NVIDIA H20，Pod 使用 `NVIDIA_VISIBLE_DEVICES=2`，进程内 `CUDA_VISIBLE_DEVICES=0` |
| Build | `/data/llama.cpp-omni-codex-phase2-20260826/build-cuda-phase2` |
| Server | `/data/llama.cpp-omni-codex-phase2-20260826/build-cuda-phase2/bin/llama-omni-server` |
| Port | `19070` |
| LLM model | `/data/omni-convert/gguf/MiniCPM-o-4_5-F16.gguf` |
| Native frontend weights | `/data/models/MiniCPM-o-4_5/assets/token2wav/speech_tokenizer_v2_25hz.onnx` 和 `campplus.onnx` |

本次请求使用 `media_type=1`、`use_tts=true`、`duplex_mode=false`，验证目标是 TTS voice cloning 和 HTTP streaming contract。两次 `omni_init` 使用不同的 `tts_ref_audio`，中间复用同一个 `19070` server：

| Voice | `tts_ref_audio` | APM `n_pos` | First Audio Response | WAV 数量 | audio token chunk 数量 |
|---|---|---:|---:|---:|---|
| voice1 | `sunwukong.wav` | 135 | 358 ms | 16 | 46, 78, 98, 86, 86 |
| voice2 | `nezha.wav` | 96 | 440 ms | 11 | 30, 79, 74, 82 |

两次请求都满足：

- `/health` 返回 `200`；
- `/v1/stream/omni_init` 返回 `{"success":true}`；
- `/v1/stream/prefill` 返回 `success=true`；
- `/v1/stream/decode` 返回完整 SSE，且每次恰好包含一个 `[DONE]`；
- 每个 voice 都生成 `generation_done.flag`；
- 所有 WAV 都是非空的 `24 kHz / mono / PCM16`；
- 两个 voice 的 audio token 和 WAV hash 不同。

原始结果目录（`20260827` 是 run directory 名称）：

```text
/data/tts-http-e2e-20260827/voice1
/data/tts-http-e2e-20260827/voice2
/data/tts-server-e2e-20260826.log
```

`PromptBundle` 是 session 初始化期间的临时数据，安装 cache 后由 C++ 删除，因此结果目录不会保留 bundle 二进制。`n_pos` 只用于记录 APM prompt 的输入规模；两个 voice 的 audio token 和 WAV hash 不同，证明两次请求产生了不同的 reference-specific TTS 结果，但这不等同于音质评测。

### Nanjing H20 WebSocket E2E

验证日期：2026-08-27。测试使用同一个 `19070` 隔离 server，历史 `23550` 服务保持运行。客户端使用 Python 标准库实现 WebSocket handshake 和 masked text frame，不依赖额外 WebSocket client 包；reference WAV 先读取为 mono PCM，再编码为 base64 float32 PCM。

| 项目 | 实际值 |
|---|---|
| Pod | `default/codex-tts-dev-20260826` |
| Node | `vm-7-13-ubuntu` |
| GPU | NVIDIA H20，测试进程内 `CUDA_VISIBLE_DEVICES=0` |
| Server | `/data/llama.cpp-omni-codex-phase2-20260826/build-cuda-phase2/bin/llama-omni-server` |
| Port | `19070` |
| Reference WAV | `/data/models/MiniCPM-o-4_5/assets/sunwukong.wav` |
| Input mode | `turn_based`，`streaming=false`，`use_tts_template=true` |
| WebSocket handshake | `HTTP/1.1 101 Switching Protocols` |
| `response.done.audio` | 非空 base64 float32 PCM，`962880` samples，按 `24 kHz` 解释为 `40.12 s` |
| `response.done` metrics | `prefill_ms=96.744`，`generate_ms=160.155`，`wall_clock_ms=5965.975`，`cost_tts_ms=5709.077` |
| `session.closed` | HTTP close 返回 `200`，WebSocket 收到 `reason=client_closed` |

完整请求链路是：

```text
session.init
  -> base64 float32 PCM 解码
  -> 临时 WAV
  -> native Token2Wav frontend
  -> Token2Mel cache setup
  -> session.created
  -> input.append
  -> C++ TTS worker
  -> response.done.audio
```

本次 `response.done` 返回的文本是“这是音色克隆的 WebSocket 端到端测试。”。输出 WAV 已解码为 `24 kHz / mono / PCM16`，RMS 为 `2.6693e-5`，peak 为 `0.00818`，不是全零数据。server log 记录了 C++ Token2Wav model 初始化、TTS worker 生成音频、`generation_done.flag` 写入和 session disconnect。

原始结果：

```text
/data/tts-ws-e2e-20260827/ws_sunwukong_response.wav
/data/tts-ws-e2e-20260827/server.log
```

另一个 session 通过 `POST /sessions/{session_id}/close` 关闭，HTTP 返回：

```json
{"closed":true,"ok":true,"session_id":"1a2fd46fa3ec7bf70220493256fcf86f"}
```

随后 WebSocket 收到 `session.closed`，reason 为 `client_closed`。测试结束后，`19070` 已停止，`23550` 仍在监听，没有残留的测试 server 或 Python Token2Wav service。

### Nanjing H20 Python/C++ TTS 对比

验证日期：2026-08-27。两条实现使用同一组 28 个 audio tokens，token 序列 SHA256 为 `4cb328714dccc1270571e480a0e3989e408565c87da8a94d4615fb19e352509b`。两条实现都使用同一个 Python StepAudio2 model directory、`n_timesteps=5` 和同一 reference WAV；C++ 侧使用 native frontend 加载 reference voice，Python 侧使用 `Token2WavService.set_ref_audio()`。

| Voice | C++ samples | Python samples | C++ RMS | Python RMS | C++ zero crossings | Python zero crossings | SHA256 是否相同 |
|---|---:|---:|---:|---:|---:|---:|---|
| `sunwukong.wav` | 20160 | 24000 | 0.08645 | 0.06660 | 2110 | 1509 | 否 |
| `nezha.wav` | 20160 | 24000 | 0.20185 | 0.19050 | 863 | 831 | 否 |

两条实现都返回非空 `24 kHz / mono / PCM16` WAV，说明 native C++ frontend、Python `Token2Wav` 和动态 reference voice 都有可运行结果。实测输出长度在两个 voice 上都不同，且 WAV SHA256 不同；因此当前证据不能支持 Python TTS 与 C++ TTS 的音频 parity 或音质等价结论。输出长度差异说明两条实现的 streaming window 输出或尾部裁剪规则还没有统一，具体原因需要后续用相同的 intermediate Mel、vocoder 输入和随机状态继续定位。

原始结果：

```text
/data/tts-python-cpp-compare-20260827/comparison.json
/data/tts-python-cpp-compare-20260827/cpp/voice1.wav
/data/tts-python-cpp-compare-20260827/cpp/voice2.wav
/data/tts-python-cpp-compare-20260827/python/voice1.wav
/data/tts-python-cpp-compare-20260827/python/voice2.wav
/data/tts-python-cpp-compare-20260827/cpp.log
/data/tts-python-cpp-compare-20260827/python-service.log
/data/tts-python-cpp-compare-20260827/python-service-voice2.log
```

## 已完成项和限制

阶段二和阶段三已完成：

- native C++ frontend 可以从 reference WAV 在线计算 PromptBundle；
- C++ Token2WavSession 可以在已加载模型上安装动态 voice cache；
- HTTP path-based `tts_ref_audio` 可以完成 native frontend、Token2Mel cache setup、TTS streaming 和 WAV 输出；
- WebSocket `/backend` 可以接收 base64 float32 `tts_ref_audio`，完成 session activation、turn-based TTS 和 `response.done.audio`；
- WebSocket session 可以通过 close primitive 返回 `session.closed`，并在关闭后复用 shared context；
- 静态 `prompt_cache.gguf` fallback 保持可用；
- C++ frontend、PromptBundle、session 和 nanjing H20 HTTP E2E 均有验证入口。

仍有以下限制：

1. Python TTS 与 C++ TTS 的输出长度和波形 hash 不一致。当前对比只证明两条实现都能生成非空结果，不证明音频 parity、音质等价或逐 sample 一致。
2. 本次 HTTP 和 WebSocket TTS 音频验证使用 simplex/turn-based 流程，没有把 full-duplex WebSocket 的持续输入、打断和客户端播放时序纳入验收。
3. voice cache 仍是 session 级状态，服务端不会跨 session 持久化或复用动态 PromptBundle。
