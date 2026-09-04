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
        std::chrono::steady_clock::time_point t_enq;
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
                // short window so concurrent sessions land in one forward; long steps (prefill) may wait
                // longer (OMNI_EXEC_PREFILL_WINDOW_US) so that a whole 1 s tick of sessions is merged,
                // but never while a decode step is pending
                {
                    bool any_long = false, any_short = false;
                    for (auto * s : queue) { (s->n > 8 ? any_long : any_short) = true; }
                    static const int prefill_window_us = getenv("OMNI_EXEC_PREFILL_WINDOW_US") ? atoi(getenv("OMNI_EXEC_PREFILL_WINDOW_US")) : 0;
                    const int w = (any_long && !any_short && prefill_window_us > window_us_) ? prefill_window_us : window_us_;
                    if (w > 0) {
                        // wait until a short step arrives or the window elapses
                        cv.wait_for(lk, std::chrono::microseconds(w), [&]{
                            if (!running) return true;
                            for (auto * s : queue) if (s->n <= 8) return true;
                            return false;
                        });
                    }
                }
                // Scheduling policy (OMNI_EXEC_POLICY):
                //   equal   : one forward = steps of EXACTLY the same token count (one per sequence). [default]
                //             The hybrid (gated-delta-net + attention) memory splits a ubatch by sequence unless
                //             every sequence contributes the same number of tokens; mixed lengths therefore turn
                //             one batch into several full forwards. Equal-length grouping keeps it to one.
                //             Longest pending length first (prefill before decode); the rest wait a round.
                //   mixed   : take everything pending (one step per sequence)           [old behaviour]
                //   decode  : decode steps (n<=8) first, then <= OMNI_EXEC_MAX_PREFILL prefills
                static const char * pol = getenv("OMNI_EXEC_POLICY");
                static const std::string policy = pol ? pol : "mixed";
                static const int max_prefill = getenv("OMNI_EXEC_MAX_PREFILL") ? atoi(getenv("OMNI_EXEC_MAX_PREFILL")) : 2;
                int tok = 0, n_long = 0;
                std::vector<bool> seq_taken(self->n_seq, false);
                if (policy == "equal") {
                    int pick_n = 0;
                    for (auto * s : queue) pick_n = std::max(pick_n, s->n);
                    for (auto it = queue.begin(); it != queue.end();) {
                        step * s = *it;
                        if (s->n != pick_n || seq_taken[s->seq] || tok + s->n > self->max_tokens) { ++it; continue; }
                        seq_taken[s->seq] = true; tok += s->n; batch.push_back(s); it = queue.erase(it);
                    }
                } else {
                    const bool decode_first = policy == "decode";
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
            }
            if (batch.empty()) continue;
            // llama.cpp's equal-split for non-unified KV only merges sequences whose ids are increasing
            // in batch order; keep the batch sorted by sequence id so one forward stays one ubatch.
            std::sort(batch.begin(), batch.end(), [](const step * a, const step * b){ return a->seq < b->seq; });
            std::lock_guard<std::mutex> rl(run_mtx);
            run_batch(batch);
        }
    }
    int window_us_ = 3000;

    void run_batch(std::vector<step *> & batch) {
        llama_context * ctx = self->ctx;
        const int n_embd = self->n_embd, n_vocab = self->n_vocab, npos = self->n_pos_per_embd;
        int n_real = 0, n_out_real = 0; for (auto * s : batch) { n_real += s->n; n_out_real += s->all_outputs ? s->n : 1; }
        // Fixed-shape padding (OMNI_EXEC_PAD=<tok granularity>[,<out granularity>]): the ggml graph shape depends on
        // the ubatch token count and on the number of output rows; every distinct shape forces a CUDA graph
        // re-capture. Pad both up to a bucket with dummy tokens on a reserved sequence (id n_seq) that is
        // cleared right after the forward, so consecutive batches of similar size hit the same captured graph.
        static const int pad_tok = getenv("OMNI_EXEC_PAD") ? atoi(getenv("OMNI_EXEC_PAD")) : 0;
        static const int pad_out = getenv("OMNI_EXEC_PAD") && strchr(getenv("OMNI_EXEC_PAD"), ',') ? atoi(strchr(getenv("OMNI_EXEC_PAD"), ',') + 1) : pad_tok;
        int n_tok = n_real, n_out_pad_rows = 0;
        if (pad_tok > 0) {
            const int n_out_tgt = ((n_out_real + pad_out - 1) / pad_out) * pad_out;
            n_out_pad_rows = n_out_tgt - n_out_real;
            n_tok = ((n_real + pad_tok - 1) / pad_tok) * pad_tok;
            while (n_tok - n_real < n_out_pad_rows) n_tok += pad_tok;   // need at least one dummy token per padded output row
            if (n_tok > self->max_tokens) { n_tok = n_real; n_out_pad_rows = 0; }
        }
        const int n_dummy = n_tok - n_real;
        // one contiguous embd buffer + pos (mrope: 4 sections, broadcast) + seq ids + output flags
        std::vector<float>        embd((size_t) n_tok * n_embd, 0.0f);
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
        for (int j = 0; j < n_dummy; j++) {   // dummy tokens: reserved seq, positions 0.., first n_out_pad_rows produce (ignored) outputs
            for (int k = 0; k < npos; k++) pos[(size_t) k * n_tok + off + j] = j;
            seq_ids[off + j] = self->n_seq; seq_ptr[off + j] = &seq_ids[off + j];
            out[off + j] = j < n_out_pad_rows ? 1 : 0;
        }
        llama_batch b = {};
        b.n_tokens = n_tok; b.embd = embd.data(); b.pos = pos.data();
        b.n_seq_id = n_seq_id.data(); b.seq_id = seq_ptr.data(); b.logits = out.data();
        const auto t0 = std::chrono::steady_clock::now();
        double qwait_max = 0, qwait_sum = 0;
        for (auto * s : batch) { double w = std::chrono::duration<double, std::milli>(t0 - s->t_enq).count(); qwait_sum += w; qwait_max = std::max(qwait_max, w); }
        self->qwait_ms_total += qwait_sum; self->qwait_ms_max = std::max(self->qwait_ms_max, qwait_max);
        const int rc = llama_decode(ctx, b);
        if (n_dummy > 0) llama_memory_seq_rm(llama_get_memory(ctx), self->n_seq, -1, -1);
        self->n_pad_tokens += n_dummy;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        self->decode_ms_total += ms;
        static const bool trace = getenv("OMNI_EXEC_TRACE") != nullptr;
        if (trace) { int mx = 0; for (auto * s : batch) mx = std::max(mx, s->n); fprintf(stderr, "[exec] steps=%zu tok=%d max_n=%d ms=%.1f qwait_avg=%.1f qwait_max=%.1f\n", batch.size(), n_tok, mx, ms, qwait_sum / batch.size(), qwait_max); }
        self->n_batches++; self->n_steps += batch.size(); self->n_tokens_total += n_tok;
        for (auto * s : batch) self->step_len_hist[s->n]++;
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
    auto * s = new impl::step(); s->seq = seq; s->pos0 = pos0; s->embd = embd; s->n = n_tokens; s->all_outputs = all_outputs; s->t_enq = std::chrono::steady_clock::now();
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
    const bool pad = getenv("OMNI_EXEC_PAD") != nullptr;
    cp.n_seq_max  = n_seq + (pad ? 1 : 0);          // +1: reserved sequence for fixed-shape padding tokens
    cp.n_ctx      = (uint32_t) n_ctx_per_seq * cp.n_seq_max;   // n_ctx must be a multiple of n_seq_max
    cp.kv_unified = getenv("OMNI_EXEC_KV_UNIFIED") != nullptr;
    cp.embeddings = true;              // hidden states for TTS
    cp.n_batch    = std::max<uint32_t>(cp.n_batch, 4096);
    if (const char * u = getenv("OMNI_EXEC_UBATCH")) { int v = atoi(u); if (v > 0) { cp.n_ubatch = v; cp.n_batch = std::max<uint32_t>(cp.n_batch, v); } }
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
