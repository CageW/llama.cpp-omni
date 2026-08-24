# Phase 3 runtime autotuning validation on Nanjing H20

## Result

Phase 3 calibration completed on 2026-08-21 using two idle NVIDIA H20 GPUs. A follow-up refresh evaluated all five bounded placement candidates. Four candidates passed the structural and real-time constraints. `gpu_cpu_hybrid` initialized Token2Wav on the CPU, but its TTS RTF was 3.905886, so the validity gate rejected it because generated audio could not keep pace with real-time playback.

The selector kept the Phase 2 `encoder_split` baseline because it had the lowest valid score, 0.337657. `token2wav_split`, the closest alternative, scored 0.338404 and was 0.22% slower by the scoring policy. The 5% baseline protection did not affect this result because no candidate beat `encoder_split`.

The HTTP integration run also generated a complete spoken response and reached the final audio callback. Paraformer ASR recovered the requested floor-count sequence from the streaming WAV chunks. The cache still records `audio_semantics_verified=false` because this check does not compare the C++ TTS output with the Python baseline required by the Phase 3 design.

## Environment

The Kubernetes Pod ran on `vm-7-13-ubuntu`. Kubernetes exposed two NVIDIA H20 GPUs as `CUDA0` and `CUDA1`; the container did not set `CUDA_VISIBLE_DEVICES`. Both devices reported 97,871 MiB total memory and 97,357 MiB free before calibration. Their UUIDs were `GPU-b4c041c3-81a4-e81f-25d9-1933452e77fb` and `GPU-3a2d8d85-db0c-29bd-281b-9bbd19972404`. The binary was compiled with CUDA 12.8.93 for compute capability 9.0.

The remote source directory is a mirror without `.git`, so the embedded commit identity is `unknown`. The five-candidate binary SHA-256 is `a3f766ac0995af4529ef7f132756b2bd5fdb9232e352f90fabc2b0493f5984ba`. The cache fingerprint includes the available build identity, device inventory, model metadata, fixture metadata, CLI overrides, and runtime inputs. On this mirror, the `unknown` commit field cannot distinguish source revisions by itself; the measured SHA-256 records the exact binary used for this report.

## Measurements

| Candidate | Valid | Score | Frame-total P95 | Encoder P95 | Decode P95 | TTFT P95 | First-audio P95 | TTS RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `consolidated` | yes | 0.342955 | 166.691 ms | 117.586 ms | 166.647 ms | 26.485 ms | 413.084 ms | 0.801674 |
| `encoder_split` | yes | 0.337657 | 166.246 ms | 111.560 ms | 166.205 ms | 25.234 ms | 391.164 ms | 0.798434 |
| `encoder_split_local_projector` | yes | 0.340624 | 169.850 ms | 110.562 ms | 169.807 ms | 25.165 ms | 394.672 ms | 0.798954 |
| `token2wav_split` | yes | 0.338404 | 166.710 ms | 109.878 ms | 166.664 ms | 25.272 ms | 394.179 ms | 0.797767 |
| `gpu_cpu_hybrid` | no | 0 | 166.597 ms | 109.820 ms | 166.549 ms | 25.327 ms | 3,950.314 ms | 3.905886 |

Every candidate completed all 12 frames, produced eight SPEAK frames, delivered one final audio callback with 227,520 finite samples, and recorded no timeout. `gpu_cpu_hybrid` still failed because structural completion does not satisfy the real-time TTS requirement.

TTS RTF is the wall time from submission of the frame that starts the SPEAK turn through the final audio callback, divided by the emitted audio duration. The runner does not resample or accelerate playback.

The selected `encoder_split` placement uses `CUDA0` for LLM, TTS, and Token2Wav. It uses `CUDA1` for Vision, Audio, and Projector. `token2wav_split` moved Token2Wav to `CUDA1`, but its score was 0.22% higher than `encoder_split`. `encoder_split_local_projector` kept Projector on `CUDA0`, but its score was 0.88% higher. During `gpu_cpu_hybrid`, the runtime logged `CPU backend using 8 threads`; Token2Wav inference took about 3.65 seconds for each one-second audio chunk, which explains the failed TTS RTF gate.

A normal server configuration lookup using the new cache reported the selected devices and `autotune_cache=applied`. The earlier HTTP integration run using the original two-candidate cache also initialized the same placement. `nvidia-smi` attributed 19,329 MiB to that server process on `CUDA0` and 2,531 MiB on `CUDA1` after initialization.

## HTTP and audio verification

`POST /v1/stream/omni_init` now performs the reserved `index=0` prefill before it returns. Before that fix, the first client frame waited indefinitely because the duplex worker threads had not started. The final run completed 28 client frames, moved from SPEAK back to LISTEN, and delivered one final audio callback.

The client deliberately sent `duplex_mode=false`, `tts_gpu_layers=100`, and `token2wav_device=cpu` while the server was running with `--profile auto`. The server kept the applied profile values: duplex mode enabled, `tts_gpu_layers=-1`, and Token2Wav on `gpu:0`. This proves that request fields cannot silently replace the placement selected by the cache.

The model's streaming text was:

```text
没问题，1楼，2楼，3楼，4楼，5楼，6楼，7楼，8楼，9楼，10楼，11楼，12楼，13楼，14楼，15楼，16楼，17楼，18楼，19楼，20楼，21楼，22楼，23楼，到24楼啦。
```

The run wrote 23 mono, 24 kHz, 16-bit WAV chunks. Per-chunk Paraformer ASR recovered the sequence from 2楼 through 24楼, while the first-floor chunk and a few chunk boundaries had recognition errors. Concatenating the chunks directly is not a valid reconstruction because the streaming windows overlap; the combined file therefore contains repeated boundary audio. The ASR result confirms that the output has the expected floor-count semantics, but it is not a bitwise, acoustic, or Python-baseline equivalence test.

The local evidence directory is `/tmp/phase3-final-20260821T045915015Z`. The combined WAV SHA-256 is `f1dd63bb807e0d945b9fd1c4d3b10e645f73e477456e21b55a97a77207b679e8`.

## Cache behavior

The five-candidate `--autotune refresh` exited with status 0 after writing `/data/omni-autotune-five-candidates.json`. It did not enter the HTTP server path. A normal `--profile auto --print-effective-config` invocation reported `autotune_cache=applied`, `resolved_profile=autotune_encoder_split`, and the six expected module placements. The original two-candidate evidence remains in `/data/omni-autotune-phase3.json`; the new binary rejects that cache with `autotune cache candidate count does not match the current candidate plan` and falls back to the deterministic Phase 2 profile.

Changing only `--ctx-size` from 8192 to 4096 made the same cache stale. The resolver reported `autotune_cache=stale` and used the deterministic Phase 2 profile. This confirms that runtime inputs participate in the fingerprint.

The cache reader recomputes validity for every candidate, including candidates recorded with `valid=false`. It rejects a record when the stored validity, score, metrics, or rejection reason disagrees with the current scoring policy. This prevents a modified non-selected candidate from changing selection behavior while avoiding validation.

The five-candidate cache SHA-256 is `39a069ee77bcfb868e29a55138bcbdb21b33dc61fe9c22d907ab41e07dacb936`. The full refresh log is `/data/omni-autotune-five-candidates-refresh.log`, and the retained invalid-candidate evidence is under `/data/autotune-runs/20260821T081719Z-5011719587854798-0/gpu_cpu_hybrid`.

## Verification

The focused local and H20 CTest suites each passed `test-head-code-layout`, `test-runtime-profile`, and `test-runtime-autotune`. The candidate-plan tests verify the exact placement of all six modules in every candidate and confirm that a machine without a secondary accelerator receives only `consolidated`. The cache tests include a regression case that marks a valid candidate invalid and confirms that the cache reader rejects the inconsistency. The runtime session tests confirm that an applied profile controls `duplex_mode`, `tts_gpu_layers`, and `token2wav_device`.

The TTS loader read the standard GGUF `head_code` tensor as output-major and reported `head_code_weight[0]=0.01361847`, `head_code_weight[1]=-0.01663208`, and `head_code_weight[768]=0.04098511`. These values match the expected tensor layout. The remaining semantic gate is the explicit comparison against the Python TTS baseline, so `audio_semantics_verified` stays `false`.

The runner waits up to 30 seconds for each frame result and up to 120 seconds for the final audio callback. Completion uses condition variables driven by the duplex and audio callbacks. A backend call already executing inside a GPU library cannot be interrupted by those condition variables, so these limits reject overdue results but do not provide a process-level hard deadline for a stuck GPU call.
