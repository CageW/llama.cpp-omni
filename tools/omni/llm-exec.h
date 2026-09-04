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

    // submit a run of embeddings for `seq` starting at `pos0`; returns when the forward is done
    omni_llm_step_result run(int seq, int pos0, const float * embd, int n_tokens, bool all_outputs);

    // memory ops are serialized with the worker (they touch the shared KV)
    bool seq_rm (int seq, int p0, int p1);
    bool seq_add(int seq, int p0, int p1, int delta);
    void seq_clear(int seq);

    // token -> input embedding (dequantized token_embd, host, shared by all sessions)
    std::vector<float> tok_emb;   // n_vocab * n_embd
    bool load_tok_emb(const std::string & llm_gguf_path);
    const float * tok_emb_row(int32_t tok) const { return tok_emb.empty() ? nullptr : tok_emb.data() + (size_t) tok * n_embd; }

    int alloc_seq();               // next free sequence id (-1 if none)
    void free_seq(int seq);

    // stats
    uint64_t n_batches = 0, n_steps = 0, n_tokens_total = 0;
    uint64_t max_steps_in_batch = 0;
    double   decode_ms_total = 0;
    double   qwait_ms_total = 0, qwait_ms_max = 0;   // time steps spent waiting in the executor queue
    std::map<int, uint64_t> step_len_hist;   // token count of submitted steps -> count

    ~omni_llm_exec();
    struct impl;
    std::unique_ptr<impl> p;
};

// Create the shared context on the model. n_ctx_per_seq * n_seq context, unified KV off,
// embeddings on (hidden states are needed for TTS).
std::unique_ptr<omni_llm_exec> omni_llm_exec_create(struct llama_model * model, const struct llama_context_params & base,
                                                    int n_seq, int n_ctx_per_seq);
