#include "llm-exec.h"
#include "omni-impl.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>

struct omni_llm_exec::impl {
    struct step {
        int seq = 0; int pos0 = 0; const float * embd = nullptr; int n = 0; bool all_outputs = false;
        std::promise<omni_llm_step_result> done;
    };
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<step *> queue;
    bool running = true;
    std::thread worker;
    omni_llm_exec * self = nullptr;
    std::mutex run_mtx;              // serializes llama_decode with memory ops on the shared ctx
    std::vector<bool> seq_used;

    void loop() {
        std::vector<step *> batch;
        while (true) {
            batch.clear();
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&]{ return !queue.empty() || !running; });
                if (!running && queue.empty()) return;
                // short window so concurrent sessions land in one forward
                if (window_us_ > 0) {
                    cv.wait_for(lk, std::chrono::microseconds(window_us_), [&]{ return !running; });
                }
                // Scheduling policy (OMNI_EXEC_POLICY):
                //   mixed   : take everything pending (one step per sequence)                      [default]
                //   decode  : decode steps (n<=8) first and alone; prefill steps (n>8) only when no decode
                //             is pending, and at most OMNI_EXEC_MAX_PREFILL (default 2) prefill steps per
                //             forward -> bounds the head-of-line delay a speaking session sees.
                static const char * pol = getenv("OMNI_EXEC_POLICY");
                static const int max_prefill = getenv("OMNI_EXEC_MAX_PREFILL") ? atoi(getenv("OMNI_EXEC_MAX_PREFILL")) : 2;
                const bool decode_first = pol && std::string(pol) == "decode";
                int tok = 0, n_long = 0;
                std::vector<bool> seq_taken(self->n_seq, false);
                bool have_short = false;
                if (decode_first) for (auto * s : queue) if (s->n <= 8) { have_short = true; break; }
                for (auto it = queue.begin(); it != queue.end();) {
                    step * s = *it;
                    const bool is_long = s->n > 8;
                    if (seq_taken[s->seq] || tok + s->n > self->max_tokens) { ++it; continue; }
                    if (decode_first) {
                        if (have_short && is_long) { ++it; continue; }
                        if (!have_short && is_long && n_long >= max_prefill) { ++it; continue; }
                    }
                    seq_taken[s->seq] = true; tok += s->n; n_long += is_long; batch.push_back(s); it = queue.erase(it);
                }
            }
            if (batch.empty()) continue;
            std::lock_guard<std::mutex> rl(run_mtx);
            run_batch(batch);
        }
    }
    int window_us_ = 3000;

    void run_batch(std::vector<step *> & batch) {
        llama_context * ctx = self->ctx;
        const int n_embd = self->n_embd, n_vocab = self->n_vocab, npos = self->n_pos_per_embd;
        int n_tok = 0; for (auto * s : batch) n_tok += s->n;
        // one contiguous embd buffer + pos (mrope: 4 sections, broadcast) + seq ids + output flags
        std::vector<float>        embd((size_t) n_tok * n_embd);
        std::vector<llama_pos>    pos((size_t) n_tok * npos);
        std::vector<int32_t>      n_seq_id(n_tok, 1);
        std::vector<llama_seq_id> seq_ids(n_tok);
        std::vector<llama_seq_id*> seq_ptr(n_tok);
        std::vector<int8_t>       out(n_tok, 0);
        int off = 0;
        for (auto * s : batch) {
            memcpy(embd.data() + (size_t) off * n_embd, s->embd, (size_t) s->n * n_embd * sizeof(float));
            for (int j = 0; j < s->n; j++) {
                for (int k = 0; k < npos; k++) pos[(size_t) k * n_tok + off + j] = s->pos0 + j;
                seq_ids[off + j] = s->seq; seq_ptr[off + j] = &seq_ids[off + j];
                out[off + j] = s->all_outputs ? 1 : (j == s->n - 1 ? 1 : 0);
            }
            off += s->n;
        }
        llama_batch b = {};
        b.n_tokens = n_tok; b.embd = embd.data(); b.pos = pos.data();
        b.n_seq_id = n_seq_id.data(); b.seq_id = seq_ptr.data(); b.logits = out.data();
        const auto t0 = std::chrono::steady_clock::now();
        const int rc = llama_decode(ctx, b);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        self->decode_ms_total += ms;
        static const bool trace = getenv("OMNI_EXEC_TRACE") != nullptr;
        if (trace) { int mx = 0; for (auto * s : batch) mx = std::max(mx, s->n); fprintf(stderr, "[exec] steps=%zu tok=%d max_n=%d ms=%.1f\n", batch.size(), n_tok, mx, ms); }
        self->n_batches++; self->n_steps += batch.size(); self->n_tokens_total += n_tok;
        self->max_steps_in_batch = std::max<uint64_t>(self->max_steps_in_batch, batch.size());
        // scatter outputs: output rows are in batch order over tokens with out[i]!=0
        int row = 0;
        for (auto * s : batch) {
            omni_llm_step_result r; r.ok = (rc == 0);
            const int n_out = s->all_outputs ? s->n : 1;
            if (r.ok) {
                r.logits.resize((size_t) n_out * n_vocab);
                r.hidden.resize((size_t) n_out * n_embd);
                for (int j = 0; j < n_out; j++) {
                    const float * lg = llama_get_logits_ith(ctx, row + j);
                    const float * hd = llama_get_embeddings_ith(ctx, row + j);
                    if (lg) memcpy(r.logits.data() + (size_t) j * n_vocab, lg, n_vocab * sizeof(float));
                    if (hd) memcpy(r.hidden.data() + (size_t) j * n_embd, hd, n_embd * sizeof(float));
                }
            }
            row += n_out;
            s->done.set_value(std::move(r));
        }
    }
};

omni_llm_step_result omni_llm_exec::run(int seq, int pos0, const float * embd, int n_tokens, bool all_outputs) {
    auto * s = new impl::step(); s->seq = seq; s->pos0 = pos0; s->embd = embd; s->n = n_tokens; s->all_outputs = all_outputs;
    auto fut = s->done.get_future();
    { std::lock_guard<std::mutex> lk(p->mtx); p->queue.push_back(s); }
    p->cv.notify_all();
    omni_llm_step_result r = fut.get();
    delete s;
    return r;
}

// memory ops: run on the worker thread to serialize with decode (queue an empty step is awkward,
// so use the mutex + a direct call; llama memory ops are not concurrent-safe with decode)
bool omni_llm_exec::seq_rm(int seq, int p0, int p1) {
    std::lock_guard<std::mutex> lk(p->run_mtx);
    return llama_memory_seq_rm(llama_get_memory(ctx), seq, p0, p1);
}
bool omni_llm_exec::seq_add(int seq, int p0, int p1, int delta) {
    std::lock_guard<std::mutex> lk(p->run_mtx);
    llama_memory_seq_add(llama_get_memory(ctx), seq, p0, p1, delta);
    return true;
}
void omni_llm_exec::seq_clear(int seq) {
    std::lock_guard<std::mutex> lk(p->run_mtx);
    llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
}
int omni_llm_exec::alloc_seq() {
    std::lock_guard<std::mutex> lk(p->mtx);
    if ((int) p->seq_used.size() < n_seq) p->seq_used.resize(n_seq, false);
    for (int i = 0; i < n_seq; i++) if (!p->seq_used[i]) { p->seq_used[i] = true; return i; }
    return -1;
}
void omni_llm_exec::free_seq(int seq) {
    std::lock_guard<std::mutex> lk(p->mtx);
    if (seq >= 0 && seq < (int) p->seq_used.size()) p->seq_used[seq] = false;
}
bool omni_llm_exec::load_tok_emb(const std::string & llm_gguf_path) {
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &ctx_meta };
    struct gguf_context * g = gguf_init_from_file(llm_gguf_path.c_str(), gp);
    if (!g) return false;
    struct ggml_tensor * t = ctx_meta ? ggml_get_tensor(ctx_meta, "token_embd.weight") : nullptr;
    bool ok = false;
    if (t && t->ne[0] == n_embd) {
        const int64_t nv = t->ne[1];
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float && t->data) {
            tok_emb.resize((size_t) nv * n_embd);
            traits->to_float(t->data, tok_emb.data(), (int64_t) nv * n_embd);
            ok = true;
        }
    }
    gguf_free(g); if (ctx_meta) ggml_free(ctx_meta);
    return ok;
}

omni_llm_exec::~omni_llm_exec() {
    if (p) {
        { std::lock_guard<std::mutex> lk(p->mtx); p->running = false; }
        p->cv.notify_all();
        if (p->worker.joinable()) p->worker.join();
    }
    if (ctx) llama_free(ctx);
}

std::unique_ptr<omni_llm_exec> omni_llm_exec_create(struct llama_model * model, const struct llama_context_params & base,
                                                    int n_seq, int n_ctx_per_seq) {
    auto e = std::make_unique<omni_llm_exec>();
    llama_context_params cp = base;
    cp.n_seq_max  = n_seq;
    cp.n_ctx      = (uint32_t) n_ctx_per_seq * n_seq;
    cp.kv_unified = getenv("OMNI_EXEC_KV_UNIFIED") != nullptr;
    cp.embeddings = true;              // hidden states for TTS
    cp.n_batch    = std::max<uint32_t>(cp.n_batch, 4096);
    e->ctx = llama_new_context_with_model(model, cp);
    if (!e->ctx) return nullptr;
    e->model = model; e->n_seq = n_seq;
    e->n_embd  = llama_n_embd(model);
    e->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const enum llama_rope_type rt = llama_model_rope_type(model);
    e->n_pos_per_embd = (rt == LLAMA_ROPE_TYPE_MROPE || rt == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
    e->max_tokens = (int) cp.n_batch;
    e->p = std::make_unique<omni_llm_exec::impl>();
    e->p->self = e.get();
    if (const char * w = getenv("OMNI_EXEC_WINDOW_US")) e->window_us = atoi(w);
    e->p->window_us_ = e->window_us;
    e->p->worker = std::thread([pi = e->p.get()]{ pi->loop(); });
    return e;
}
