#pragma once
// Shared, batched LLM executor for multi-session omni.
//
// One llama_context (n_seq_max = N sessions) and one worker thread per GPU. Sessions submit
// "steps" (a run of input embeddings for their sequence at a given position); the worker
// merges all steps pending within a short window into a single llama_decode() and hands back
// the logits / hidden states each step asked for. Cross-session batching amortizes the
// per-forward cost (the launch-bound MoE prefill) that otherwise scales linearly with sessions.
#include "llama.h"

#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <map>

struct omni_llm_step_result {
    bool ok = false;
    std::vector<float> logits;   // n_out * n_vocab (n_out = n_tokens if all_outputs, else 1)
    std::vector<float> hidden;   // n_out * n_embd   (same layout)
};

struct omni_llm_exec {
    struct llama_model *   model = nullptr;
    struct llama_context * ctx   = nullptr;
    int n_seq      = 0;
    int n_embd     = 0;
    int n_vocab    = 0;
    int n_pos_per_embd = 1;
    int window_us  = 3000;        // how long the worker waits for more sessions to join a batch
    int max_tokens = 4096;        // logical batch cap
    // scheduling knobs (per executor; filled from OMNI_EXEC_* or OMNI_<PREFIX>_EXEC_* at create time)
    std::string policy = "mixed";  // none | static | mixed | equal | decode | pd | ripe | adaptive
    // The three settings the batching question is actually about:
    //   none     : one step per forward, never wait. The lower bound, so the value of batching is measurable.
    //   static   : wait a fixed window (and/or for a fixed step count) read at startup, then take everything.
    //   adaptive : wait only for sequences predicted to come back, with the budget following GPU pressure.
    // `mixed` is not "no batching": it waits up to window_us but returns the moment any decode step is
    // queued, which under this workload is almost always, so it batches whatever happens to be pending.
    int long_threshold = 8;        // a step with n > long_threshold counts as "prefill" for scheduling
    int prefill_target = 0;        // ripe/gather: release prefills once this many are queued (0 = n_seq)
    int prefill_deadline_us = 200000;
    int prefill_gather_us = 0, prefill_window_us = 0, max_prefill = 2;
    bool trace = false;
    int pad_steps = 0;             // fixed-shape padding (OMNI_EXEC_PAD)
    // adaptive policy (OMNI_EXEC_POLICY=adaptive). A session's forwards form a chain: after its result is
    // delivered it resubmits the next step after a short host-side gap (sampling), so the executor can predict
    // WHEN the sequences it just served will be back. Holding Δ for k predicted arrivals saves k forwards' fixed
    // cost of GPU time and costs Δ of latency for each step already queued; GPU time is only worth something when
    // the GPU is the bottleneck, so the saving is weighted by the busy fraction. The worker holds only while
    // Δ < k·F·busy·hold_gain, re-evaluated every time a step lands. Idle GPU -> no hold -> pure latency mode.
    //   hold_gain      : weight of saved GPU time vs added latency         (OMNI_EXEC_HOLD_GAIN, default 1.5)
    //   hold_max_us    : total hold bound per queued step, and the ETA horizon   (OMNI_EXEC_HOLD_MAX_US, default 20000)
    //   chain_gap_max_us: a resubmit later than this is a frame boundary, not a chain link, and is not predicted
    //                                                                      (OMNI_EXEC_CHAIN_GAP_MAX_US, default 60000)
    //   fixed_cost_us  : initial estimate of the per-forward fixed cost; refined online from 1-token batches
    //                                                                      (OMNI_EXEC_FIXED_COST_US, default 6500)
    //   batch_target_ms: when busy, cap prefill tokens per forward so one forward stays near this (OMNI_EXEC_BATCH_TARGET_MS,
    //                    default 0 = off; it measured worse at the ceiling). Decode steps are taken first; at least one prefill always goes.
    //   cap_min_tokens : floor of that cap (OMNI_EXEC_CAP_MIN_TOKENS, default 32)
    //   cap_busy_threshold: busy fraction above which the cap engages (OMNI_EXEC_CAP_BUSY_THRESHOLD, default 0.5)
    double hold_gain = 1.5;
    int hold_max_us = 20000;
    int batch_target_ms = 0;
    int cap_min_tokens = 32;
    double cap_busy_threshold = 0.5;
    int chain_gap_max_us = 60000;
    int fixed_cost_us = 6500;
    // static policy: the fixed window and the fixed step-count trigger, whichever comes first
    //   static_window_us : OMNI_EXEC_STATIC_WINDOW_US, default 10000 (10 ms)
    //   static_target    : OMNI_EXEC_STATIC_TARGET, default 0 = window only
    int static_window_us = 10000;
    int static_target = 0;

    // submit a run of embeddings for `seq` starting at `pos0`; returns when the forward is done
    omni_llm_step_result run(int seq, int pos0, const float * embd, int n_tokens, bool all_outputs);

    // memory ops are serialized with the worker (they touch the shared KV)
    bool seq_rm (int seq, int p0, int p1);
    bool seq_add(int seq, int p0, int p1, int delta);
    void seq_clear(int seq);

    // token -> input embedding (dequantized token_embd, host, shared by all sessions)
    std::vector<float> tok_emb;   // n_vocab * n_embd
    std::vector<ggml_backend_dev_t> devices;   // GPU(s) pinned for this executor (kept alive; CP.devices points here)
    bool load_tok_emb(const std::string & llm_gguf_path);
    const float * tok_emb_row(int32_t tok) const { return tok_emb.empty() ? nullptr : tok_emb.data() + (size_t) tok * n_embd; }

    int alloc_seq();               // next free sequence id (-1 if none)
    void free_seq(int seq);

    // stats
    uint64_t n_batches = 0, n_steps = 0, n_tokens_total = 0, n_pad_tokens = 0;
    uint64_t n_forward = 0;   // omni: actual graph computes (ubatches) executed; one logical batch can split into many
    uint64_t max_steps_in_batch = 0;
    double   decode_ms_total = 0;   // wall time inside llama_decode + device sync (what the GPU was actually busy for)
    double   qwait_ms_total = 0, qwait_ms_max = 0;   // time steps spent waiting in the executor queue
    double   hold_ms_total = 0;     // adaptive: time the worker deliberately held a non-empty queue waiting for predicted arrivals
    uint64_t n_hold = 0, n_hold_hit = 0;   // adaptive: holds started / holds where at least one predicted sequence arrived
    double   busy_fraction = 0;     // EMA of decode time over wall time, the GPU pressure signal the adaptive policy uses
    double   fixed_cost_ms = 6.5;   // online estimate of the per-forward fixed cost (from 1-token batches)
    std::map<int, uint64_t> step_len_hist;   // token count of submitted steps -> count
    std::map<int, uint64_t> batch_width_hist; // steps per batch -> count (the batching width actually achieved)

    // thread-safe copy of the counters above (for /health and reports)
    struct snapshot_t {
        uint64_t n_batches, n_steps, n_tokens_total, n_forward, max_steps_in_batch, n_hold, n_hold_hit;
        double decode_ms_total, qwait_ms_total, qwait_ms_max, hold_ms_total, busy_fraction, fixed_cost_ms;
        std::map<int, uint64_t> batch_width_hist;
    };
    snapshot_t snapshot() const;

    ~omni_llm_exec();
    struct impl;
    std::unique_ptr<impl> p;
};

// Create the shared context on the model. n_ctx_per_seq * n_seq context, unified KV off,
// embeddings on (hidden states are needed for TTS).
// `device` (optional): name of a single GPU (e.g. "CUDA1") to pin this executor to; nullptr/empty uses the
// default device list from `base`. Used for the multi-worker layout (one executor per GPU).
// `env_prefix` (optional, e.g. "OMNI_TTS"): read scheduling knobs from <prefix>_EXEC_* instead of OMNI_EXEC_*,
// so a second executor (the TTS model) can be tuned independently of the LLM executor.
// omni EP lock-step: the N LLM executors (one per card, experts shared/split) run their forwards in rounds. A round is
// started by an executor with ripe work; every executor joins with its pending steps of the round's step length
// (or with nothing), the number of participants is published to the CUDA backend, and the participants' forwards
// meet at every expert op for one merged grouped GEMM per card. Call before creating the executors.
void omni_llm_exec_lockstep_init(int n_members);

std::unique_ptr<omni_llm_exec> omni_llm_exec_create(struct llama_model * model, const struct llama_context_params & base,
                                                    int n_seq, int n_ctx_per_seq, const char * device = nullptr,
                                                    const char * env_prefix = nullptr);
