# Phase 4 runtime autotuning validation on Nanjing H20

## Result

Phase 4 calibration completed on 2026-08-21 with two idle NVIDIA H20 GPUs. The refresh measured 21 rounds across seven GPU candidates and screened four Token2Wav CPU candidates. It exited with status 0 after 750.786 seconds and selected the F16 `encoder_split` candidate.

A normal server startup using the selected cache also passed an HTTP end-to-end regression. The client completed 28 paced prefill/decode calls, received every SSE completion marker, and generated two spoken turns. Paraformer ASR recovered the expected sentence from both turns.

All five F16 placement candidates produced the same normalized response and TTS token hashes in every paired round. The Q8_0 and Q4_K_M candidates were valid performance measurements, but their hashes differed from the F16 baseline. They remained advisory because the run did not pass `--autotune-allow-quantized`. All four CPU candidates were rejected during screening because their TTS RTF exceeded the 1.5 limit.

The selected cache passed a normal `--print-effective-config` lookup. Changing only `ctx_size` from 8192 to 4096 made the cache stale and restored the deterministic Phase 2 profile. The ASR semantic smoke test passed, but `audio_semantics_verified` remains `false` because the run did not compare C++ TTS output with the Python TTS baseline.

## Environment

The worker hostname was `VM-7-13-ubuntu`. Physical GPU2 and GPU3 were exposed inside the validation container as `CUDA0` and `CUDA1`:

| Container device | Physical UUID | Model | Total memory | Free memory before calibration |
|---|---|---|---:|---:|
| `CUDA0` | `GPU-4078f7ad-9a3e-1e72-1a88-152993e2a3dd` | NVIDIA H20 | 97,356 MiB | 97,029 MiB |
| `CUDA1` | `GPU-52c52f72-13de-b893-867c-7b343f5f2786` | NVIDIA H20 | 97,356 MiB | 97,029 MiB |

The build used `minicpm-o45-demo:local`, CUDA 12.8.93, GNU 11.4, Release mode, and compute capability 9.0. The source and build directories were isolated from the existing service:

```text
/data/llama.cpp-omni-phase4-20260821
/data/llama.cpp-omni-phase4-20260821/build-cuda-phase4
```

The refresh used this server command inside the container:

```bash
/workspace/build-cuda-phase4/bin/llama-omni-server \
    --profile auto \
    --model-dir /data/omni-convert/gguf-phase4-20260821 \
    --autotune refresh \
    --autotune-rounds 3 \
    --autotune-warmup-rounds 1 \
    --autotune-cache /data/omni-autotune-phase4.json
```

The run used scoring policy `2`, one complete warmup turn before each measured round, the fixed 12-frame duplex fixture, and CPU thread probes `8,16,32,64`.

## Aggregate results

Each metric below is `median + MAD` across three measured rounds. Lower score is better. Frame, Encoder, Decode, TTFT, and First audio values are milliseconds.

| Candidate | Role | Score | Frame P95 | Encoder P95 | Decode P95 | TTFT P95 | First audio P95 | TTS RTF | Output result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `precision_q8_0` | advisory precision | 0.426804 | 175.765 | 103.808 | 175.717 | 26.298 | 367.837 | 0.745761 | valid, F16 hash mismatch |
| `precision_q4_k_m` | advisory precision | 0.358714 | 161.474 | 101.315 | 161.414 | 27.452 | 367.878 | 0.575026 | valid, F16 hash mismatch |
| `consolidated` | F16 placement | 0.428718 | 181.006 | 105.037 | 180.951 | 25.083 | 374.899 | 0.747341 | valid, F16 hash match |
| `encoder_split` | F16 baseline | 0.428544 | 181.423 | 105.215 | 181.363 | 25.093 | 373.659 | 0.747577 | selected, F16 hash match |
| `encoder_split_local_projector` | F16 placement | 0.426809 | 178.597 | 101.879 | 178.516 | 25.040 | 371.847 | 0.746120 | valid, F16 hash match |
| `token2wav_split` | F16 placement | 0.426920 | 180.173 | 104.392 | 180.104 | 25.026 | 371.248 | 0.745534 | valid, F16 hash match |
| `output_pipeline_split` | F16 placement | 0.425985 | 179.553 | 104.164 | 179.472 | 25.189 | 367.575 | 0.745364 | valid, F16 hash match |

`output_pipeline_split` had the lowest F16 aggregate score, 0.60 percent below `encoder_split`. Its round 1 and round 2 scores were higher than the paired `encoder_split` scores, and the aggregate improvement was below the 5 percent selection threshold. The selector therefore kept `encoder_split`, with this recorded reason:

```text
kept the Phase 2 baseline because a lower aggregate score was worse in at least one paired round
```

Q4_K_M reduced the aggregate score by 16.3 percent relative to the F16 baseline and used much less GPU memory. Its generated response and TTS token sequence changed. The run recorded the performance result but did not authorize Q4_K_M for normal startup.

## Raw measured rounds

The table contains every scored round. Frame, Encoder, Decode, TTFT, and First audio values are milliseconds.

| Candidate | Round | Score | Frame P95 | Encoder P95 | Decode P95 | TTFT P95 | First audio P95 | TTS RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `precision_q8_0` | 1 | 0.422163 | 168.289 | 101.456 | 168.231 | 25.664 | 360.406 | 0.740929 |
| `precision_q8_0` | 2 | 0.430355 | 175.765 | 121.564 | 175.717 | 26.298 | 367.837 | 0.745761 |
| `precision_q8_0` | 3 | 0.425262 | 172.763 | 102.632 | 172.699 | 26.207 | 365.361 | 0.744179 |
| `precision_q4_k_m` | 1 | 0.353825 | 156.869 | 98.802 | 156.818 | 26.802 | 359.999 | 0.569959 |
| `precision_q4_k_m` | 2 | 0.358840 | 161.474 | 102.461 | 161.414 | 27.279 | 367.878 | 0.575026 |
| `precision_q4_k_m` | 3 | 0.357525 | 159.591 | 100.059 | 159.530 | 27.452 | 365.280 | 0.574306 |
| `consolidated` | 1 | 0.422417 | 173.584 | 98.546 | 173.540 | 24.612 | 365.129 | 0.741645 |
| `consolidated` | 2 | 0.426739 | 178.793 | 102.328 | 178.734 | 25.016 | 370.767 | 0.746431 |
| `consolidated` | 3 | 0.428718 | 181.006 | 105.037 | 180.951 | 25.083 | 374.899 | 0.747341 |
| `encoder_split` | 1 | 0.422246 | 173.901 | 98.627 | 173.854 | 24.613 | 364.411 | 0.741625 |
| `encoder_split` | 2 | 0.426152 | 178.410 | 102.428 | 178.343 | 25.047 | 370.141 | 0.745257 |
| `encoder_split` | 3 | 0.428544 | 181.423 | 105.215 | 181.363 | 25.093 | 373.659 | 0.747577 |
| `encoder_split_local_projector` | 1 | 0.424046 | 175.005 | 101.120 | 174.947 | 24.680 | 368.313 | 0.742339 |
| `encoder_split_local_projector` | 2 | 0.427177 | 178.636 | 103.226 | 178.576 | 25.040 | 372.240 | 0.746120 |
| `encoder_split_local_projector` | 3 | 0.425650 | 176.801 | 100.361 | 176.732 | 24.973 | 370.080 | 0.745185 |
| `token2wav_split` | 1 | 0.426095 | 178.400 | 102.504 | 178.333 | 25.020 | 370.243 | 0.745053 |
| `token2wav_split` | 2 | 0.428260 | 181.473 | 106.449 | 181.431 | 25.015 | 374.201 | 0.746027 |
| `token2wav_split` | 3 | 0.425288 | 176.626 | 100.616 | 176.563 | 25.044 | 369.237 | 0.744572 |
| `output_pipeline_split` | 1 | 0.425014 | 178.175 | 102.645 | 178.099 | 25.007 | 366.483 | 0.744650 |
| `output_pipeline_split` | 2 | 0.427325 | 182.019 | 106.194 | 181.947 | 25.131 | 370.094 | 0.746208 |
| `output_pipeline_split` | 3 | 0.424260 | 176.798 | 101.126 | 176.726 | 25.189 | 365.390 | 0.743935 |

Every measured round initialized all modules, completed 12 frames without a timeout, emitted finite audio, and delivered two final audio callbacks. The F16 rounds produced 140 TTS tokens and 145,920 audio samples.

## CPU screening

The CPU candidates moved Token2Wav to CPU and kept LLM, Vision, Audio, TTS, and Projector on the same GPU placement as `encoder_split_local_projector`.

| Candidate | Threads | Encoder P95 | Decode P95 | TTFT P95 | First audio P95 | TTS RTF | CUDA0/CUDA1 peak | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `gpu_cpu_hybrid_t8` | 8 | 144.670 | 185.856 | 26.378 | 3,982.526 | 3.910610 | 17,480 / 2,306 MiB | rejected |
| `gpu_cpu_hybrid_t16` | 16 | 110.172 | 168.157 | 25.188 | 3,845.422 | 3.791381 | 17,434 / 2,264 MiB | rejected |
| `gpu_cpu_hybrid_t32` | 32 | 109.398 | 169.836 | 25.189 | 3,793.927 | 3.726768 | 17,434 / 2,264 MiB | rejected |
| `gpu_cpu_hybrid_t64` | 64 | 109.187 | 169.068 | 25.168 | 3,759.911 | 3.690085 | 17,434 / 2,264 MiB | rejected |

Increasing the thread count from 8 to 64 reduced TTS RTF by 5.6 percent. The fastest CPU result still took 3.69 seconds to produce one second of audio, so none of the CPU candidates entered the measured-round schedule.

## Output consistency

The five F16 placement candidates shared these aggregate hashes and token count:

```text
response_sha256=9f6c9f24153233e9962f9df32800b9907191c190db37135d8d5b43e1fd89c591
tts_tokens_sha256=d6d696140471cdbca4d2b13829d1d6f867496a71ee7b15ec5c59d257349ddd5f
tts_token_count=140
```

The advisory precision candidates produced stable hashes across their own three rounds:

| Candidate | Response SHA-256 | TTS token SHA-256 | TTS tokens | Matches F16 baseline |
|---|---|---|---:|---|
| `precision_q8_0` | `56673c61fce2beac7c9312ad2dc7e862608c78a032a903822c7c3833ade732dd` | `c279debbc8d1361e5eb7eea51c4ad1a77abc08f0696900c9672442af25d8a421` | 140 | no |
| `precision_q4_k_m` | `de0af7cb2113caa476adf55b1482fafe981fed00af13e73223f869366da91f7b` | `9e7c2ea35986b89533b7dc643b3ef2560be2730a8f0483af58da5db65d78286d` | 120 | no |

The repeated hashes show that each precision produced the same output across its three runs with this fixture and seed. They do not establish spoken-audio semantics or acoustic equivalence.

## Memory estimates and peaks

The cache records static estimates and the highest observed device allocation across the measured rounds. CPU candidates have screening peaks because screening rejected them before measured rounds.

| Candidate | Estimated memory | Observed peak |
|---|---|---|
| `precision_q8_0` | CUDA0 15,134.0 MiB; CUDA1 3,649.0 MiB | CUDA0 12,390.0 MiB; CUDA1 2,318.0 MiB |
| `precision_q4_k_m` | CUDA0 11,272.4 MiB; CUDA1 3,649.0 MiB | CUDA0 9,092.0 MiB; CUDA1 2,280.0 MiB |
| `consolidated` | CUDA0 26,836.7 MiB | CUDA0 21,422.0 MiB |
| `encoder_split` | CUDA0 23,187.6 MiB; CUDA1 3,649.0 MiB | CUDA0 19,138.0 MiB; CUDA1 2,280.0 MiB |
| `encoder_split_local_projector` | CUDA0 23,459.3 MiB; CUDA1 3,377.4 MiB | CUDA0 19,154.0 MiB; CUDA1 2,264.0 MiB |
| `token2wav_split` | CUDA0 21,200.5 MiB; CUDA1 5,636.2 MiB | CUDA0 17,420.0 MiB; CUDA1 4,046.0 MiB |
| `output_pipeline_split` | CUDA0 19,236.5 MiB; CUDA1 7,600.2 MiB | CUDA0 15,966.0 MiB; CUDA1 5,454.0 MiB |

The first refresh had no previous matching cache, so `memory_estimate_inputs` is empty. Each candidate still contains its static estimate and observed memory samples. A later matching refresh can populate `memory_estimate_inputs` from the previous measured module deltas.

## Applied cache and stale cache

Normal lookup with the same runtime inputs exited with status 0 and reported:

```text
resolved_profile=autotune_encoder_split
autotune_cache=applied
autotune_cache_reason=autotune cache fingerprint matched
autotune_selected_candidate=encoder_split
output_consistent_with_baseline=true
audio_semantics_verified=false
```

The applied placement was:

| Module | Device | Precision |
|---|---|---|
| LLM | `CUDA0` | F16 |
| Vision | `CUDA1` | F16 |
| Audio | `CUDA1` | F16 |
| TTS | `CUDA0` | F16 |
| Projector | `CUDA1` | F16 |
| Token2Wav | `CUDA0` (`gpu:0`) | F16 |

The same command with `--ctx-size 4096` reported `autotune_cache=stale` and `autotune cache fingerprint does not match the current runtime`. It used `full_omni_multi_accelerator_f16` with `ctx_size=4096`.

## HTTP end-to-end regression

The Phase 4 binary started a normal HTTP server on port 19060 with `/data/omni-autotune-phase4.json`. Startup reported `resolved_profile=autotune_encoder_split`, `autotune_cache=applied`, and the six placements listed above.

The `omni_init` request deliberately asked for `duplex_mode=false`, `tts_gpu_layers=100`, `token2wav_device=cpu`, and `token2wav_threads=64`. The applied Runtime Profile kept duplex mode enabled, used `tts_gpu_layers=-1`, and ran Token2Wav on `gpu:0` with 8 threads. Request fields therefore did not replace the measured placement.

The valid run sent 28 one-second fixture frames at a 1.0 second cadence. It completed in 33.586 seconds, returned 28 SSE `[DONE]` markers, and moved from SPEAK back to LISTEN twice. The two turns produced this text:

```text
没问题，现在到了21楼，还有三楼就到24楼了。
没问题，现在到了 21楼，还有三楼就到 24楼了。
```

Token2Wav wrote 14 mono, 24 kHz, 16-bit PCM WAV chunks. The client contract plays those chunks in numeric order. Paraformer processed each complete turn after the corresponding chunks were joined in that order:

| Turn | WAV files | Playback duration | Playback SHA-256 | Paraformer result |
|---|---|---:|---|---|
| 1 | `wav_0.wav` through `wav_6.wav` | 6.72 s | `e4029f7a84dd9ae5e6359d28a5a47c942b921e9ba808fe63a6b8e35978827157` | `没问题现在到了二十一楼还有三楼就到二十四楼了` |
| 2 | `wav_7.wav` through `wav_13.wav` | 6.40 s | `843f2c7bed57f9c23124b7c1aed41b41b0785803f220c4f90d1f0501bf179db4` | `没问题现在到了二十一楼还有三楼就到二十四楼了` |

An earlier diagnostic sent the same 28 one-second frames in 8.8 seconds. That rate violated the documented 1 Hz client cadence, so its output was excluded from the verdict. Running ASR on isolated one-second WAV chunks also produced boundary errors; ASR on the ordered playback for each turn recovered the complete sentence.

The end-to-end evidence is under `/data/omni-phase4-e2e-20260821`. The final verdict is `/data/omni-phase4-e2e-20260821/e2e-verdict.json`, with SHA-256 `fd81e9b0a11c61e85fab2188d2d68a6aafe4da7970b4957cfcac038bc5162614`.

This test covers the HTTP path, cache enforcement, streaming completion, WAV format, and spoken semantics for the fixture. It does not compare C++ TTS output with the Python TTS baseline, so the cache still records `audio_semantics_verified=false`.

## Artifacts

The model directory contains an F16 baseline plus Q8_0 and Q4_K_M files generated directly from that F16 file:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `MiniCPM-o-4_5-F16.gguf` | 16,384,959,136 | `d1e6984531bab1962d8bc73da4b6dffc5c2d9b0da336603943df04100e57c3de` |
| `MiniCPM-o-4_5-Q8_0.gguf` | 8,707,877,536 | `ae6af22ad3b7f1a7bf667922af84b9eb3e2199cc86f402702eaf9b054054788d` |
| `MiniCPM-o-4_5-Q4_K_M.gguf` | 5,026,714,400 | `3a6bc1bcef3d9ea9503cb765a26c153fbc79ffa38df573b1760ffbb36b97f3a0` |

The persistent validation evidence is:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `/data/omni-autotune-phase4.json` | 247,276 | `4c5ea538ffd30df8cfe804f2d2d849e984bdd25d1820bed02b0775fcd214df65` |
| `/data/omni-autotune-phase4-refresh.log` | 1,960,908 | `87bdc2b6214aab515977e0e2d51e294f9ddfae5bd0ce0fa6c10eb76a175e8267` |
| `llama-omni-server` | 1,356,352 | `4c054f227e2600e77db887959a8fe89eb553d152ede52f05b204bb07708a144e` |
| `libggml-cuda.so` | 62,548,640 | `9984c5ee623ea7a819975a057502b6ddd3f3548cd80d5ecd3c49a64e671d00e3` |

## Verification and cleanup

The focused H20 CTest run passed all four Phase 4 targets:

```text
test-omni-server-runtime-profile
test-head-code-layout
test-runtime-profile
test-runtime-autotune
```

The same four focused tests passed in the local build. The local full CTest run passed 52 of 61 tests. The remaining failures were outside the Phase 4 targets: the MTMD include path, a GGML mixed-type assertion, and external Hugging Face download or dependency failures. A full local build also stopped at 99 percent on the MTMD `mtmd.h` include path; the Phase 4 server and test targets built successfully both locally and on H20.

After calibration, both cache lookups, and the HTTP end-to-end regression, physical GPU2 and GPU3 reported 0 MiB used. No Phase 4 container remained. The existing service on the other GPUs was left running.
