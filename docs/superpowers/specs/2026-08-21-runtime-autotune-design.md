# Phase 3 runtime autotuning design

## Status

This design was approved on 2026-08-21. Phase 3 adds an explicit calibration mode and a versioned local cache. It does not run a benchmark during normal server startup.

## Scope

Phase 1 added deterministic `--profile auto` resolution. Phase 2 added explicit module placement and connected the resolved devices to the LLM, Vision, Audio, TTS, Projector, and Token2Wav initialization paths.

Phase 3 measures five bounded placement plans that the current runtime can execute safely:

- The consolidated candidate places every module on the primary accelerator.
- The encoder-split candidate keeps the LLM, TTS, and Token2Wav on the primary accelerator, and places Vision, Audio, and Projector on the secondary accelerator.
- The encoder-split-local-projector candidate keeps Projector with the LLM, TTS, and Token2Wav on the primary accelerator, and places Vision and Audio on the secondary accelerator.
- The Token2Wav-split candidate keeps the LLM and TTS on the primary accelerator, and places Vision, Audio, Projector, and Token2Wav on the secondary accelerator.
- The GPU-CPU-hybrid candidate keeps the LLM, TTS, and Projector on the primary accelerator, places Vision and Audio on the secondary accelerator, and runs Token2Wav on CPU.

Phase 3 does not enumerate arbitrary module permutations, change the selected LLM quantization, move TTS away from the LLM device, or distribute modules across physical machines. Those changes would expand the state space and require separate correctness work.

## Command behavior

Calibration uses the existing server binary:

```bash
llama-omni-server \
    --profile auto \
    --model-dir /path/to/MiniCPM-o-4_5-gguf \
    --autotune refresh
```

The command loads each eligible candidate, runs the fixed duplex fixture, writes the selected result to the autotune cache, and exits. It does not open the HTTP port.

A normal server invocation keeps the current startup behavior:

```bash
llama-omni-server \
    --profile auto \
    --model-dir /path/to/MiniCPM-o-4_5-gguf \
    --host 0.0.0.0 \
    --port 9060
```

The resolver reads a matching cache entry before finalizing the effective runtime configuration. A missing, stale, or invalid cache entry does not block startup; the resolver uses the Phase 2 deterministic placement and prints the reason that the cache was not used.

`--autotune-cache <path>` overrides the default cache path. Without an override, the implementation uses `fs_get_cache_file()` so it follows the existing `LLAMA_CACHE` and platform cache-directory behavior.

## Components

The new `runtime-autotune` module owns calibration-specific behavior. It builds fingerprints, creates eligible candidates, runs a supplied calibration callback, scores completed records, and reads or writes the cache. Fingerprint, cache, and scoring functions remain independent of model execution so their behavior can be covered by fast unit tests.

`runtime-profile` remains responsible for hardware detection, model inventory, deterministic fallback, and the final `effective_runtime_config`. It may apply a cache result only after `runtime-autotune` confirms that the entry matches the current process and resolved inputs.

The calibration runner reuses the existing duplex frame API and timing hooks used by `perf-duplex`. It must not invoke `perf-duplex` as a child process or duplicate its request loop in a script. Metrics needed by both tools should move into a small shared C++ data structure while model lifecycle stays in the caller.

The server entry point handles CLI mode selection. `--autotune refresh` runs calibration and exits. Normal mode resolves the profile, optionally applies a cache result, then starts the HTTP server as it does today.

## Fingerprint and cache

An autotune result is reusable only when all inputs that can change placement behavior still match. The fingerprint contains:

- The cache schema version, `llama_build_number()`, `llama_commit()`, and the build target.
- The GGML backend type, device name, device description, and total memory for every detected device.
- The selected model paths plus file size and modification time for every model used by the candidate.
- The requested profile, explicit CLI overrides, sampling settings, CPU settings, cache types, tensor overrides, memory-mapping flags, and environment variables that can change model loading or execution.
- File metadata for the first 12 audio/image pairs in `duplex_omni_test_case`, plus scoring-policy version `1`.

The first version uses model file metadata instead of hashing complete GGUF files. Hashing the full model tree would add substantial I/O before every startup. A replaced file invalidates the cache when its size or modification time changes.

The cache is versioned JSON and stores the fingerprint, each candidate's effective configuration, raw measurements, validity result, rejection reason, selected candidate, and creation time. The writer creates a temporary file in the destination directory, flushes and closes it, then uses an atomic rename. A truncated JSON file, unknown schema version, fingerprint mismatch, or selected device that is no longer present makes the entry unusable.

## Candidate execution

The Phase 2 resolver produces the starting configuration. Runtime autotuning derives at most five candidates from it:

1. The consolidated candidate is eligible when the primary accelerator has enough free memory for all active modules.
2. The four split candidates are eligible when a secondary accelerator exists, has at least 4 GiB free, and every requested backend device passes capability validation. The GPU-CPU-hybrid candidate uses the same GPU eligibility rule because Vision and Audio still run on the secondary accelerator.

Each candidate gets a new Omni context and an independent run directory below the autotune cache directory. State from one candidate must not be reused by another candidate. Calibration uses sampler seed `42` and the first 12 frames from the checked-in `duplex_omni_test_case` fixture. If the fixture ends during a SPEAK turn, the runner appends an explicit end-of-turn marker to the TTS queue and waits for the resulting final audio callback. Successful run directories are removed after their measurements have been written to the cache; failed run directories are retained for diagnosis.

A candidate is invalid when any frame times out, a requested module cannot initialize on its selected device, the fixture does not produce a SPEAK transition, Token2Wav emits non-finite samples, or the audio stream ends without `is_final=true`. Calibration disables normal Projector and Token2Wav device fallback so the measurements describe the candidate recorded in the cache. Invalid candidates remain in the cache report with a concrete rejection reason but cannot be selected.

## Metrics and scoring

The calibration record contains frame-total P95, encoder P95, decode P95, TTFT, first-audio latency, TTS RTF, completed frame count, SPEAK count, audio sample count, and timeout count. TTS RTF uses the wall time from submission of the frame that starts a SPEAK turn through the final audio callback, divided by the emitted audio duration. This conservative interval includes the LLM decision and TTS queueing; it does not change the playback rate or resample the audio.

Selection first applies two real-time constraints: frame-total P95 must stay below the 1000 ms fixture input interval, and TTS RTF must stay below 1.0. Scoring policy `1` uses this formula for candidates that satisfy both constraints:

```text
score = 0.35 * frame_total_p95_ms / 1000
      + 0.25 * decode_p95_ms / 1000
      + 0.20 * first_audio_p95_ms / 1000
      + 0.20 * tts_rtf
```

When the lower score improves on the Phase 2 deterministic candidate by less than 5 percent, the selector keeps the Phase 2 placement. This prevents a marginal benchmark difference from introducing cross-device traffic. If every candidate is invalid, calibration returns a non-zero exit code, preserves the diagnostic report, and does not replace an existing valid cache entry.

## Correctness boundary

Calibration checks whether the duplex and audio pipelines complete structurally. It cannot determine whether generated speech carries the intended semantics.

A follow-up Nanjing run fixed the TTS `head_code` tensor layout, reached the final audio callback, and produced 24 kHz streaming WAV chunks. Paraformer ASR recovered the expected floor-count sequence from those chunks, with errors at the overlapping chunk boundaries. Phase 3 must still record `audio_semantics_verified=false` until the C++ TTS path is compared with the Python baseline. The calibration result is performance and structural-completion evidence; the separate ASR run is semantic evidence, but neither establishes Python-baseline equivalence.

## Failure handling

A cache read, parse, or fingerprint error alone does not fail normal server startup. The effective configuration prints one of these outcomes with a specific reason: cache applied, cache missing, cache stale, cache invalid, or cache bypassed by explicit overrides.

Calibration failures are stricter. An invalid CLI combination, unwritable cache path, model initialization failure, no valid candidate, or incomplete fixture returns a non-zero exit code. The implementation retains the failed candidate's run directory and leaves the previous valid cache untouched.

The calibration runner uses condition-based completion from the duplex and audio callbacks. It does not use an arbitrary sleep to decide that TTS has finished.

## Tests

Implementation follows TDD. Tests are added in this order:

1. Fingerprint tests prove that device, model metadata, build identity, fixture identity, and explicit override changes invalidate a cache entry.
2. Cache tests cover a valid round trip, unknown schema versions, malformed JSON, an unavailable selected device, and atomic replacement behavior.
3. Scoring tests cover one valid candidate, competing candidates, the tie threshold, real-time constraint failures, and the case where every candidate is invalid.
4. The CLI smoke test proves that `--autotune refresh` enters calibration mode and exits without opening the HTTP port. Cache unit tests reject stale or inconsistent records, and the Nanjing validation confirms that normal `--profile auto` applies a matching cache entry.
5. The existing runtime-profile and server smoke tests continue to pass without a cache.

The remote validation uses the Nanjing H20 node with two free physical GPUs exposed to the process. It runs all five candidates, checks that the selected placement matches the cache, and compares the actual per-GPU memory allocation with the selected module devices. The report must retain the TTS semantic limitation described above.

## Delivery order

The implementation can be reviewed in four slices: pure fingerprint and cache code, pure candidate scoring, calibration runner and CLI mode, then Nanjing validation. Production behavior does not change until the normal resolver can read a fully validated cache entry.
