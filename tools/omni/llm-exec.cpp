#include "llm-exec.h"
#include "omni-impl.h"
#include "gguf.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>

struct omni_exec_lockstep {
    std::mutex m;
    std::condition_variable cv;
    int n_members = 0, arrived = 0, with_work = 0;
    int round_len = 0;          // step length (tokens) admitted in the requested round
    uint64_t round = 0;
    bool requested = false;
    std::vector<struct omni_llm_exec::impl *> members;
};
static omni_exec_lockstep * g_lockstep = nullptr;
void omni_llm_exec_lockstep_init(int n_members) {
    g_lockstep = new omni_exec_lockstep();
    g_lockstep->n_members = n_members;
}

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
    uint64_t forward_last = 0;       // last-read graph compute count of this executor's context (per executor)
    bool lockstep_member = false;    // EP lock-step: only the LLM worker executors take part (not the TTS executor)

    void loop() {
        std::vector<step *> batch;
        while (true) {
            batch.clear();
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&]{ return !queue.empty() || !running || (lockstep_member && g_lockstep->requested); });
                if (!running && queue.empty()) return;
                if (lockstep_member) {
                    // ---- EP lock-step round: decide/join a round, take only steps of the round's length
                    const int LT = self->long_threshold;
                    int L = 0;
                    { std::lock_guard<std::mutex> g(g_lockstep->m); if (g_lockstep->requested) L = g_lockstep->round_len; }
                    if (L == 0 && !queue.empty()) {
                        // ripe rule on the own queue: decodes are never held, prefills wait for >= target or the deadline
                        const int target = self->prefill_target > 0 ? self->prefill_target : self->n_seq;
                        const auto now = std::chrono::steady_clock::now();
                        int n_long_q = 0, long_len = 0, short_len = 0; bool old = false;
                        for (auto * s : queue) {
                            if (s->n > LT) {
                                n_long_q++; if (!long_len) long_len = s->n;
                                if (std::chrono::duration_cast<std::chrono::microseconds>(now - s->t_enq).count() >= self->prefill_deadline_us) old = true;
                            } else if (!short_len) { short_len = s->n; }
                        }
                        if (short_len) { L = short_len; } else if (n_long_q >= target || old) { L = long_len; }
                        if (L == 0) {
                            // prefills not ripe yet: wait a little, or until a round is requested
                            cv.wait_for(lk, std::chrono::microseconds(std::min(self->prefill_deadline_us, 20000)),
                                        [&]{ return !running || g_lockstep->requested; });
                            continue;
                        }
                        std::vector<omni_llm_exec::impl *> members;
                        {
                            std::lock_guard<std::mutex> g(g_lockstep->m);
                            if (!g_lockstep->requested) { g_lockstep->requested = true; g_lockstep->round_len = L; } else { L = g_lockstep->round_len; }
                            members = g_lockstep->members;
                        }
                        for (auto * m : members) { if (m != this) { m->cv.notify_all(); } }
                    }
                    if (L == 0) { continue; }
                    int tok = 0;
                    std::vector<bool> seq_taken(self->n_seq, false);
                    for (auto it = queue.begin(); it != queue.end();) {
                        step * s = *it;
                        if (s->n != L || seq_taken[s->seq] || tok + s->n > self->max_tokens) { ++it; continue; }
                        seq_taken[s->seq] = true; tok += s->n; batch.push_back(s); it = queue.erase(it);
                    }
                    std::sort(batch.begin(), batch.end(), [](const step * a, const step * b){ return a->seq < b->seq; });
                    lk.unlock();
                    // barrier: the last member to arrive publishes the participant count and releases the round
                    {
                        std::unique_lock<std::mutex> g(g_lockstep->m);
                        g_lockstep->arrived++;
                        g_lockstep->with_work += batch.empty() ? 0 : 1;
                        if (g_lockstep->arrived == g_lockstep->n_members) {
                            ggml_backend_cuda_esplit_set_participants(g_lockstep->with_work);
                            if (self->trace) fprintf(stderr, "[lockstep] round %llu len %d participants %d\n", (unsigned long long) g_lockstep->round + 1, g_lockstep->round_len, g_lockstep->with_work);
                            g_lockstep->arrived = 0; g_lockstep->with_work = 0; g_lockstep->requested = false; g_lockstep->round++;
                            g_lockstep->cv.notify_all();
                        } else {
                            const uint64_t r = g_lockstep->round;
                            g_lockstep->cv.wait(g, [&]{ return g_lockstep->round != r; });
                        }
                    }
                    if (self->trace) fprintf(stderr, "[lockstep] dev-exec %p round %llu: %zu steps of len %d\n", (void *) this, (unsigned long long) g_lockstep->round, batch.size(), L);
                    if (batch.empty()) { continue; }
                    goto lockstep_run;
                }
                // short window so concurrent sessions land in one forward; long steps (prefill) may wait
                // longer (OMNI_EXEC_PREFILL_WINDOW_US) so that a whole 1 s tick of sessions is merged,
                // but never while a decode step is pending
                {
                    const int LT = self->long_threshold;
                    bool any_long = false, any_short = false;
                    for (auto * s : queue) { (s->n > LT ? any_long : any_short) = true; }
                    const bool ripe = self->policy == "ripe";
                    if (ripe) {
                        // never hold decode steps; hold prefills until >= prefill_target are queued or the oldest has waited
                        // >= prefill_deadline_us (breaks the circular wait of gather mode: a session submits its next prefill
                        // only after this frame's decodes finish, and those decodes sat in this queue behind the gather wait)
                        const int target = self->prefill_target > 0 ? self->prefill_target : self->n_seq;
                        auto ready = [&]{
                            if (!running) return true;
                            int n_long_r = 0; bool any_s = false, old = false;
                            const auto now = std::chrono::steady_clock::now();
                            for (auto * s : queue) {
                                if (s->n <= LT) { any_s = true; continue; }
                                n_long_r++;
                                if (std::chrono::duration_cast<std::chrono::microseconds>(now - s->t_enq).count() >= self->prefill_deadline_us) old = true;
                            }
                            return any_s || n_long_r >= target || old;
                        };
                        cv.wait_for(lk, std::chrono::microseconds(std::min(self->prefill_deadline_us, 20000)), ready);
                        if (running) cv.wait_for(lk, std::chrono::microseconds(window_us_), [&]{ return !running; });   // let same-tick decodes coalesce
                    } else if (self->prefill_gather_us > 0 && any_long) {
                        const int target = self->prefill_target > 0 ? self->prefill_target : self->n_seq;
                        cv.wait_for(lk, std::chrono::microseconds(self->prefill_gather_us), [&]{
                            if (!running) return true;
                            int n_long = 0;
                            for (auto * s : queue) n_long += (s->n > LT);
                            return n_long >= target;
                        });
                    } else {
                        const int w = (any_long && !any_short && self->prefill_window_us > window_us_) ? self->prefill_window_us : window_us_;
                        if (w > 0) {
                            cv.wait_for(lk, std::chrono::microseconds(w), [&]{
                                if (!running) return true;
                                for (auto * s : queue) if (s->n <= LT) return true;
                                return false;
                            });
                        }
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
                //   pd      : prefill and decode never mix in one forward. Each loop picks ONE homogeneous
                //             batch (all long, or all short); the two alternate via a toggle so neither
                //             starves. Because a batch now has a single step length, the hybrid memory's
                //             split_equal stays a single ubatch instead of exploding one mixed batch into
                //             many forwards (which is what blows up the per-batch time at high concurrency).
                const std::string & policy = self->policy;
                const int max_prefill = self->max_prefill;
                const int LT = self->long_threshold;
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
                } else if (policy == "ripe") {
                    const int target = self->prefill_target > 0 ? self->prefill_target : self->n_seq;
                    int n_long_q = 0; bool old = false;
                    const auto now = std::chrono::steady_clock::now();
                    for (auto * s : queue) if (s->n > LT) { n_long_q++; if (std::chrono::duration_cast<std::chrono::microseconds>(now - s->t_enq).count() >= self->prefill_deadline_us) old = true; }
                    const bool take_long = n_long_q >= target || old;   // prefills ripe -> one wide prefill batch; else serve decodes
                    for (auto it = queue.begin(); it != queue.end();) {
                        step * s = *it;
                        if ((s->n > LT) != take_long || seq_taken[s->seq] || tok + s->n > self->max_tokens) { ++it; continue; }
                        seq_taken[s->seq] = true; tok += s->n; batch.push_back(s); it = queue.erase(it);
                    }
                } else if (policy == "pd") {
                    // toggle: on odd iterations serve long (prefill) steps first, on even ones short (decode).
                    static bool pd_prefill_turn = true;
                    const bool want_long = pd_prefill_turn;
                    pd_prefill_turn = !pd_prefill_turn;
                    bool have_wanted = false;
                    for (auto * s : queue) if ((s->n > LT) == want_long) { have_wanted = true; break; }
                    for (auto it = queue.begin(); it != queue.end();) {
                        step * s = *it;
                        const bool is_long = s->n > LT;
                        if (is_long != want_long) { ++it; continue; }   // only take the wanted class this round
                        if (seq_taken[s->seq] || tok + s->n > self->max_tokens) { ++it; continue; }
                        seq_taken[s->seq] = true; tok += s->n; batch.push_back(s); it = queue.erase(it);
                    }
                    if (batch.empty() && have_wanted) {
                        // wanted class present but too large to fit; fall back to the other class this round
                        const bool other = !want_long;
                        for (auto it = queue.begin(); it != queue.end();) {
                            step * s = *it;
                            if ((s->n > LT) != other || seq_taken[s->seq] || tok + s->n > self->max_tokens) { ++it; continue; }
                            seq_taken[s->seq] = true; tok += s->n; batch.push_back(s); it = queue.erase(it);
                        }
                    }
                } else {
                    const bool decode_first = policy == "decode";
                    bool have_short = false;
                    if (decode_first) for (auto * s : queue) if (s->n <= LT) { have_short = true; break; }
                    for (auto it = queue.begin(); it != queue.end();) {
                        step * s = *it;
                        const bool is_long = s->n > LT;
                        if (seq_taken[s->seq] || tok + s->n > self->max_tokens) { ++it; continue; }
                        if (decode_first) {
                            if (have_short && is_long) { ++it; continue; }
                            if (!have_short && is_long && n_long >= max_prefill) { ++it; continue; }
                        }
                        seq_taken[s->seq] = true; tok += s->n; n_long += is_long; batch.push_back(s); it = queue.erase(it);
                    }
                }
            }
            lockstep_run:
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
        int n_real = 0; for (auto * s : batch) n_real += s->n;
        // Fixed-shape padding (OMNI_EXEC_PAD=<step granularity>): the hybrid memory splits a batch into ubatches with
        // split_equal, which takes one token from every participating sequence per ubatch. So (a) the graph shape of a
        // decode ubatch is set by the NUMBER OF SEQUENCES in it, and (b) dummy tokens stacked on ONE reserved sequence
        // become their own extra ubatch (a second forward). Pad therefore with 1-token steps on distinct reserved
        // sequences (ids n_seq .. n_seq+pad-1), bringing the step count up to a multiple of the granularity: the
        // decode ubatch then always has the same n_seqs / n_outputs and the captured CUDA graph is reused.
        const int pad_steps = self->pad_steps;
        int n_dummy = 0;
        if (pad_steps > 0) {
            const int n_steps_tgt = ((int) batch.size() + pad_steps - 1) / pad_steps * pad_steps;
            n_dummy = std::min(n_steps_tgt - (int) batch.size(), pad_steps);
            if (n_real + n_dummy > self->max_tokens) n_dummy = 0;
        }
        const int n_tok = n_real + n_dummy;
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
        for (int j = 0; j < n_dummy; j++) {   // dummy steps: one token each on reserved seqs n_seq+j at pos 0, output on (ignored)
            for (int k = 0; k < npos; k++) pos[(size_t) k * n_tok + off + j] = 0;
            seq_ids[off + j] = self->n_seq + j; seq_ptr[off + j] = &seq_ids[off + j];
            out[off + j] = 1;
        }
        llama_batch b = {};
        b.n_tokens = n_tok; b.embd = embd.data(); b.pos = pos.data();
        b.n_seq_id = n_seq_id.data(); b.seq_id = seq_ptr.data(); b.logits = out.data();
        const auto t0 = std::chrono::steady_clock::now();
        double qwait_max = 0, qwait_sum = 0;
        for (auto * s : batch) { double w = std::chrono::duration<double, std::milli>(t0 - s->t_enq).count(); qwait_sum += w; qwait_max = std::max(qwait_max, w); }
        self->qwait_ms_total += qwait_sum; self->qwait_ms_max = std::max(self->qwait_ms_max, qwait_max);
        const int rc = llama_decode(ctx, b);
        if (rc != 0) fprintf(stderr, "[exec] llama_decode rc=%d (steps=%zu tok=%d)\n", rc, batch.size(), n_tok);
        self->n_forward += llama_context_graph_compute_count(ctx) - forward_last; forward_last = llama_context_graph_compute_count(ctx);
        for (int j = 0; j < n_dummy; j++) llama_memory_seq_rm(llama_get_memory(ctx), self->n_seq + j, -1, -1);
        self->n_pad_tokens += n_dummy;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        self->decode_ms_total += ms;
        const bool trace = self->trace;
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
                                                    int n_seq, int n_ctx_per_seq, const char * device, const char * env_prefix) {
    auto e = std::make_unique<omni_llm_exec>();
    // knob lookup: <prefix>_EXEC_<NAME>, falling back to OMNI_EXEC_<NAME> when no prefix is given
    const std::string pfx = env_prefix && env_prefix[0] ? env_prefix : "OMNI";
    auto knob = [&](const char * name) -> const char * { std::string k = pfx + "_EXEC_" + name; return getenv(k.c_str()); };
    auto knob_i = [&](const char * name, int dflt) { const char * v = knob(name); return v ? atoi(v) : dflt; };
    llama_context_params cp = base;
    const int pad = knob_i("PAD", 0);
    e->pad_steps = pad;
    cp.n_seq_max  = n_seq + std::max(pad, 0);       // reserved sequences for fixed-shape padding steps
    cp.n_ctx      = (uint32_t) n_ctx_per_seq * cp.n_seq_max;   // n_ctx must be a multiple of n_seq_max
    cp.kv_unified = knob("KV_UNIFIED") != nullptr;
    cp.embeddings = true;              // hidden states for TTS
    cp.n_batch    = std::max<uint32_t>(cp.n_batch, 4096);
    if (const char * u = knob("UBATCH")) { int v = atoi(u); if (v > 0) { cp.n_ubatch = v; cp.n_batch = std::max<uint32_t>(cp.n_batch, v); } }
    if (const char * v = knob("POLICY")) e->policy = v;
    e->long_threshold      = knob_i("LONG_THRESHOLD", 8);
    e->prefill_target      = knob_i("PREFILL_TARGET", 0);
    e->prefill_deadline_us = knob_i("PREFILL_DEADLINE_US", 200000);
    e->prefill_gather_us   = knob_i("PREFILL_GATHER_US", 0);
    e->prefill_window_us   = knob_i("PREFILL_WINDOW_US", 0);
    e->max_prefill         = knob_i("MAX_PREFILL", 2);
    e->trace               = knob("TRACE") != nullptr;
    if (const char * w = knob("WINDOW_US")) e->window_us = atoi(w);
    if (device && device[0]) {
        // NOTE: a llama_context cannot be pinned to a GPU independently of its model: device placement is fixed
        // at llama_model load (llama_model_params.devices), and every context shares the model's device set.
        // A per-GPU executor therefore needs its own model copy (= the multi-process layout). Kept as a no-op.
        LOG_WRN("omni_llm_exec_create: per-executor device '%s' ignored (context inherits the model's devices)\n", device);
    }
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
    e->p->window_us_ = e->window_us;
    if (g_lockstep && (env_prefix == nullptr || env_prefix[0] == 0)) {
        std::lock_guard<std::mutex> g(g_lockstep->m);
        g_lockstep->members.push_back(e->p.get());
        e->p->lockstep_member = true;
    }
    e->p->worker = std::thread([pi = e->p.get()]{ pi->loop(); });
    return e;
}
