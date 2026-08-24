# Phase 4 runtime autotuning design

## Status

The behavior in this document was approved in chat on 2026-08-21. Local implementation, focused local and H20 tests, Nanjing H20 calibration, applied-cache verification, stale-cache verification, HTTP end-to-end regression, and cleanup are complete. The measured results are in the [Phase 4 Nanjing H20 validation report](../validation/2026-08-21-runtime-autotune-phase4-nanjing-h20.md).

## Problem

Phase 3 measures one run of each placement candidate in a fixed order. The Nanjing H20 result selected `encoder_split`, but the closest alternatives differed by less than one percent. One run cannot separate a small placement effect from GPU clock, cache, or scheduling noise.

Phase 3 also uses one 4 GiB secondary-device gate, fixed Token2Wav CPU threads, hard-coded LLM memory thresholds, and a cache fingerprint that cannot distinguish binaries when the source mirror reports commit `unknown`. Its score gives 60 percent of the weight to Frame-total P95 and Decode P95 even though those values were nearly identical in the H20 run.

Phase 4 makes the measurement repeatable before expanding the search. It then adds one output-pipeline placement, advisory LLM quantization measurements, CPU thread probes, per-module memory accounting, a stronger fingerprint, and scoring policy `2`.

## Safety boundary

Moving the same GGUF model between two identical CUDA devices is a low-risk change. It uses the same weights and backend implementation, although floating-point execution is not guaranteed to be bitwise identical. Moving Token2Wav between CUDA and CPU has more numerical risk because the backend kernels differ. Changing the LLM from F16 to Q8_0 or Q4_K_M can change generated text and TTS tokens.

Phase 4 does not add an audio semantics gate. It uses an output-consistency check for same-precision placement candidates:

- The runner concatenates response text chunks in callback order, converts CRLF to LF, trims leading and trailing ASCII whitespace, and hashes the remaining UTF-8 bytes with SHA-256.
- The runner serializes every emitted TTS audio token as a signed 32-bit little-endian value in emission order, appends one signed `-1` end marker, and hashes the byte sequence with SHA-256.
- A selectable placement candidate must match the Phase 2 baseline hashes in every paired measured round.
- The check detects output drift. It does not prove that matching output has correct semantics.

Quantized candidates record the same hashes and whether they match the Phase 2 baseline. A hash mismatch does not invalidate an advisory quantized result because a different quantization can legitimately produce different text. Quantized candidates cannot be selected unless the operator passes `--autotune-allow-quantized`.

Arbitrary module permutations, LLM tensor parallel, LLM layer splitting, remote execution, ASR, and Python-baseline comparison remain out of scope.

## Command behavior

The existing refresh command remains the entry point:

```bash
llama-omni-server \
    --profile auto \
    --model-dir /path/to/MiniCPM-o-4_5-gguf \
    --autotune refresh
```

Phase 4 adds these options:

| Option | Default | Behavior |
|---|---:|---|
| `--autotune-rounds N` | 3 | Runs 1 to 10 measured rounds for each candidate. The value participates in the cache fingerprint. |
| `--autotune-warmup-rounds N` | 1 | Runs 0 to 3 unscored turns before each measured round. The value participates in the cache fingerprint. |
| `--autotune-allow-quantized` | false | Allows a valid Q8_0 or Q4_K_M advisory candidate to participate in final selection. |
| `--token2wav-threads N` | 8 | Sets Token2Wav CPU threads for normal execution. The value must be positive and participates in the cache fingerprint. |

An applied Runtime Profile owns `token2wav_threads` in the same way that it owns `duplex_mode`, `tts_gpu_layers`, and `token2wav_device`. An `omni_init` request cannot replace the cached value. Legacy startup without `--profile` keeps its current behavior.

Normal server startup never runs calibration. A missing, stale, or invalid Phase 4 cache falls back to the Phase 2 deterministic profile and prints the exact reason.

## Search structure

Calibration uses a two-stage bounded search. Candidate configurations are deduplicated before execution, so a Stage A baseline that is identical to a Stage B placement runs once.

### Stage A: precision probe

Stage A uses the Phase 2 placement and varies only the LLM GGUF:

- F16, when the inventory contains an eligible F16 model.
- Q8_0, when the inventory contains an eligible Q8_0 model.
- Q4_K_M, when the inventory contains an eligible Q4_K_M model.

Vision, Audio, TTS, Projector, and Token2Wav stay on their Phase 2 model files and devices. An explicit `--model` disables the precision probe because the operator has already selected the LLM file.

The Phase 2 precision is the selectable baseline. Other precision records are advisory unless `--autotune-allow-quantized` is present. A precision candidate that exceeds the primary-device memory estimate is skipped with the required and available byte counts.

### Stage B: runtime candidates

Stage B uses the Phase 2 precision. On a dual-accelerator machine it considers these GPU placements:

| Candidate | CUDA0 | CUDA1 |
|---|---|---|
| `consolidated` | LLM, Vision, Audio, TTS, Projector, Token2Wav | none |
| `encoder_split` | LLM, TTS, Token2Wav | Vision, Audio, Projector |
| `encoder_split_local_projector` | LLM, TTS, Projector, Token2Wav | Vision, Audio |
| `token2wav_split` | LLM, TTS | Vision, Audio, Projector, Token2Wav |
| `output_pipeline_split` | LLM | Vision, Audio, TTS, Projector, Token2Wav |

Stage B also derives `gpu_cpu_hybrid_t8`, `gpu_cpu_hybrid_t16`, `gpu_cpu_hybrid_t32`, and `gpu_cpu_hybrid_t64` from the current GPU-CPU placement. Those candidates use CUDA0 for LLM, TTS, and Projector, CUDA1 for Vision and Audio, and CPU for Token2Wav. A thread value above `std::thread::hardware_concurrency()` is omitted. Duplicate values are removed.

Explicit per-module device overrides remain fixed in every candidate. Candidate generation removes configurations that become identical after applying overrides. A machine without a secondary accelerator receives only candidates that reference its primary accelerator or CPU.

## Round schedule

Each measured round creates a new Omni context. The context uses seed `42`, strict runtime placement, and a private run directory. It executes the configured number of complete unscored warmup turns, resets the duplex session and sampler state, then executes one measured turn using the 12-frame fixture.

Candidate order rotates by round. If the ordered candidates are `A, B, C`, the schedules are `A, B, C`, then `B, C, A`, then `C, A, B`. This keeps the schedule deterministic while distributing fixed-order clock and cache effects.

The runner retains raw measurements from every measured round. A candidate must complete every configured measured round to be valid. Successful run directories are removed after the cache is written. A failed candidate keeps the run directory for its first failed round.

CPU candidates have one screening exception. If a warmup turn fails structurally or reports TTS RTF greater than or equal to 1.5, the candidate is rejected immediately with `screening_rejected=true`. The cache records its warmup metrics and does not claim that measured rounds completed. This prevents an obviously non-real-time CPU path from adding several minutes to calibration.

## Memory eligibility and observation

Phase 4 replaces the 20, 13, and 9 GiB LLM thresholds and the shared 4 GiB split gate with per-module estimates. The first refresh uses these estimates:

| Module | Initial accelerator estimate |
|---|---:|
| LLM | `1.10 * model_file_bytes + 2 GiB` |
| Vision | `1.10 * model_file_bytes + 1 GiB` |
| Audio | `1.10 * model_file_bytes + 512 MiB` |
| TTS | `1.10 * model_file_bytes + 1 GiB` |
| Projector | `1.10 * model_file_bytes + 256 MiB` |
| Token2Wav | `1.10 * sum(model_directory_file_bytes) + 1 GiB` |

The planner sums estimates for the modules assigned to each accelerator. A non-baseline candidate is eligible when every accelerator keeps at least 10 percent of its currently free memory after the estimate. CPU placement does not consume accelerator headroom. The Phase 2 baseline stays in the plan even when the estimate is pessimistic; an initialization failure still makes it invalid.

An optional memory observer reads free and total device memory before model loading, after each module initializes, and at the end of the measured turn. The cache stores non-negative per-module deltas and the peak used bytes for each device. Lazy allocation can make a module delta incomplete, so the per-device peak is authoritative.

A later refresh may use measured values from an otherwise matching previous cache. Its estimate for a module is the greater of the static estimate and `1.10 * previous_measured_delta`. A changed executable, backend, driver, model, device, or runtime input prevents reuse of those measurements.

Memory is an eligibility constraint and a reported metric. It is not part of the latency score.

## Metrics and scoring policy 2

Each round records Frame-total P95, Encoder P95, Decode P95, TTFT P95, First-audio latency, TTS RTF, frame and SPEAK counts, final callback count, audio sample count, timeout count, output hashes, and device peak memory.

For every scored latency metric, the aggregate is:

```text
robust_metric = median(round_values) + median(abs(round_value - median(round_values)))
```

The second term is MAD, or median absolute deviation. It penalizes a candidate whose typical result is fast but unstable.

Scoring policy `2` uses:

```text
score = 0.20 * robust_encoder_p95_ms / 1000
      + 0.15 * robust_ttft_p95_ms / 250
      + 0.25 * robust_first_audio_p95_ms / 1000
      + 0.40 * robust_tts_rtf
```

Decode P95 remains in the report but does not receive a second score weight. Frame-total P95 is a deadline gate. Every measured round must satisfy all of these conditions:

- All 12 fixture frames complete and no frame times out.
- Frame-total P95 is below the 1000 ms fixture interval.
- At least one SPEAK transition occurs.
- TTS RTF is below 1.0.
- Token2Wav produces finite samples and at least one final audio callback.
- A same-precision placement candidate matches the baseline response-text and TTS-token hashes for the paired round.

The per-round score uses the same formula without MAD. A selectable challenger replaces the Phase 2 baseline only when it is no worse than the baseline in every paired round and its aggregate score is at least 5 percent lower. If those conditions are not met, the selector keeps the baseline.

When `--autotune-allow-quantized` is absent, advisory precision records cannot be selected. When the flag is present, a structurally valid quantized candidate may participate even when its output hashes differ. The cache and effective-config output mark that condition as `output_consistent_with_baseline=false` so the operator can see the accepted quality risk.

If the Phase 2 baseline cannot produce a valid measured round, calibration cannot establish the output reference or paired score. Refresh fails, preserves the previous cache, and retains the baseline diagnostic directory.

## Fingerprint and cache schema

Phase 4 increments the cache schema and scoring-policy versions. The fingerprint retains all Phase 3 fields and adds:

- SHA-256 of the running executable.
- Sorted identities for dynamically loaded GGML backend libraries. Each identity contains the canonical path, file size, modification time, and SHA-256. A statically linked backend is covered by the executable hash.
- CUDA driver version from the CUDA driver API when CUDA is compiled in. Other backends record their available runtime version string. An unavailable value is stored as the literal `unavailable`.
- `autotune_rounds`, `autotune_warmup_rounds`, `autotune_allow_quantized`, the CPU thread probe set, and the output-hash policy version.
- `token2wav_threads` in the effective runtime configuration and explicit overrides.

The implementation reuses the repository's SHA-256 code and hashes files as a stream. It does not invoke `sha256sum` or another child process.

The cache stores the candidate plan, round schedule, raw round records, aggregate metrics, score, selection eligibility, advisory state, output consistency, memory estimates, observed memory, rejection reason, and selected candidate. Cache validation rebuilds the current plan and recomputes round validity, aggregate metrics, scores, and selection. A mismatch makes the cache invalid.

Old Phase 3 caches are invalid under the Phase 4 schema. Normal startup falls back to Phase 2 instead of modifying or deleting the old file.

## Component changes

`runtime-profile` owns model inventory, per-module static memory estimates, hardware and driver identity, `token2wav_threads`, and the deterministic fallback.

`runtime-autotune` owns Stage A and Stage B plan generation, round scheduling, output-hash policy, robust aggregation, scoring policy `2`, cache serialization, cache validation, and selection.

`runtime-autotune-runner` owns context lifecycle, warmup and measured turns, telemetry capture, device-memory snapshots, and CPU screening. Model execution remains behind the existing Omni APIs.

The Omni runtime exposes narrow telemetry hooks for response text, TTS audio tokens, and module-initialization boundaries. The hooks are disabled during normal serving and must not copy PCM data unless calibration requests it.

Token2Wav passes `token2wav_threads` through `Token2WavSession`, `Token2Wav`, `Token2Mel`, and the vocoder CPU backend. GPU behavior is unchanged by the thread value.

The server entry point parses the new CLI options, starts calibration for `--autotune refresh`, and prints the new cache and output-consistency fields in `--print-effective-config`.

## Failure behavior

An unavailable quantized model or an ineligible non-baseline candidate produces a skip record and does not fail the refresh. A candidate initialization error, output mismatch, timeout, real-time failure, or CPU screening rejection produces an invalid candidate record with a concrete reason.

Refresh returns a nonzero status and preserves the previous cache when the Phase 2 baseline is invalid, every selectable candidate is invalid, the destination cannot be written atomically, or the completed record fails self-validation. Advisory candidate failures alone do not fail refresh.

The existing condition-variable timeouts reject overdue results but cannot interrupt a backend call already blocked inside a GPU library. Process-level worker isolation remains outside Phase 4.

## Tests

Implementation follows TDD. Tests are added in this order:

1. CLI and runtime-config tests cover valid ranges, invalid values, `token2wav_threads`, request override protection, and legacy startup.
2. Candidate-plan tests cover both stages, model availability, explicit overrides, deduplication, single-accelerator behavior, CPU thread caps, output-pipeline placement, and per-device memory rejection.
3. Schedule tests cover deterministic rotation, warmup exclusion, three measured rounds, CPU screening, and retained failure paths.
4. Output-consistency tests cover text normalization, little-endian token serialization, hash mismatch rejection for placement candidates, and advisory quantized mismatches.
5. Scoring tests cover median, MAD, policy `2`, per-round paired comparison, the 5 percent threshold, baseline failure, and recomputation during cache validation.
6. Fingerprint tests cover executable content, backend library identity, driver version, protocol inputs, model metadata, and explicit overrides.
7. Cache tests cover the new round records, memory records, atomic replacement, malformed data, Phase 3 rejection, and an unavailable selected device.
8. Integration tests use a fake calibration callback to verify stage order and failure handling without loading model weights. Existing runtime-profile, head-code-layout, server smoke, and Phase 3 regression tests continue to pass.

## Nanjing H20 validation

The remote validation builds a fresh CUDA binary and runs the complete Phase 4 refresh on the same two H20 devices used for Phase 3. The report records total calibration time, every raw round, aggregate metrics, output hashes, memory estimates and peaks, skip or rejection reasons, advisory quantization results, and the selected candidate.

Validation then starts `--print-effective-config` with the new cache and checks every module device, LLM model, Token2Wav thread count, cache status, selected candidate, and output-consistency field. A second invocation with one fingerprint input changed must reject the cache. The temporary validation container is deleted after persistent cache, log, and binary-hash evidence is verified on the worker.

The validation completed on 2026-08-21. It selected `encoder_split`, applied the matching cache, rejected the same cache after `ctx_size` changed, and passed an HTTP end-to-end regression with two ASR-verified spoken turns. Both validation GPUs returned to 0 MiB used. Q8_0 and Q4_K_M remained advisory because their output hashes differed from the F16 baseline. The report keeps `audio_semantics_verified=false` because the end-to-end regression does not compare C++ TTS output with the Python TTS baseline.

## Delivery order

Implementation is split into four reviewable slices:

1. Schema, fingerprint, CLI, runtime config, and Token2Wav thread plumbing.
2. Candidate generation, memory eligibility, output hashing, and telemetry hooks.
3. Round scheduling, robust aggregation, scoring policy `2`, selection, and cache validation.
4. H20 calibration, applied-cache verification, documentation, and cleanup.

Production startup behavior changes only after a complete Phase 4 cache passes validation. The Phase 2 deterministic profile remains the fallback throughout implementation.
