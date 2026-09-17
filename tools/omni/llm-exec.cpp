#include "llm-exec.h"
#include "omni-impl.h"
#include "gguf.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
    mutable std::mutex stats_mtx;    // guards the public counters while snapshot() copies them

    // ---- adaptive policy state (touched only by the worker thread) ----
    struct seq_pred {
        std::chrono::steady_clock::time_point t_deliver{};   // when this sequence's last result was handed back
        double gap_ema_us = 0;     // EMA of (resubmit - deliver) for chain links (short gaps)
        double gap_var_us = 0;     // EMA of |deviation|, used as the wait margin
        bool   in_flight = false;  // delivered and not yet resubmitted: a candidate to wait for
        bool   was_long = false;   // the step just delivered was a prefill (its follow-up is the frame's first decode)
    };
    std::vector<seq_pred> pred;
    std::chrono::steady_clock::time_point t_last_batch_end{};
    double busy_ema = 0;           // decode_ms / (decode_ms + idle_ms), EMA over batches
    double fixed_cost_ema_us = 0;  // 1-token single-step batch cost, EMA (the amortizable part of a forward)

    // called by the worker just before a step's promise is fulfilled; takes mtx because submitters update
    // the same table (note_submitted) under mtx, and the worker's hold decision reads it under mtx too
    void note_delivered(const step * s, std::chrono::steady_clock::time_point now) {
        if ((size_t) s->seq >= pred.size()) return;
        std::lock_guard<std::mutex> lk(mtx);
        auto & p = pred[s->seq];
        p.t_deliver = now; p.in_flight = true; p.was_long = s->n > self->long_threshold;
    }
    void note_submitted(const step * s) {
        if ((size_t) s->seq >= pred.size()) return;
        auto & p = pred[s->seq];
        if (!p.in_flight) return;
        p.in_flight = false;
        const double gap = std::chrono::duration<double, std::micro>(s->t_enq - p.t_deliver).count();
        if (gap > self->chain_gap_max_us) return;   // frame boundary (waited for the client), not a chain link
        if (p.gap_ema_us <= 0) { p.gap_ema_us = gap; p.gap_var_us = gap * 0.5; return; }
        const double dev = std::fabs(gap - p.gap_ema_us);
        p.gap_ema_us = 0.7 * p.gap_ema_us + 0.3 * gap;
        p.gap_var_us = 0.7 * p.gap_var_us + 0.3 * dev;
    }
    // adaptive: how long (us) the worker should hold the current queue for sequences predicted to resubmit.
    // 0 = dispatch now. Called with mtx held.
    //
    // Holding Δ for k predicted arrivals saves k forwards' fixed cost (k·F of GPU time). When the GPU is the
    // bottleneck every active chain queues behind it, so each saved forward is F of latency for all of them,
    // not just for the k that arrive; the steps already queued pay Δ. Hold iff Δ < k·F·busy·gain. busy is the
    // GPU pressure (0 when idle -> never hold), gain absorbs the active/queued ratio (OMNI_EXEC_HOLD_GAIN).
    int adaptive_hold_us(std::chrono::steady_clock::time_point now, size_t q_steps, int * n_expected_out) {
        const double fixed_us = fixed_cost_ema_us > 0 ? fixed_cost_ema_us : (double) self->fixed_cost_us;
        (void) q_steps;
        // candidates: in-flight sequences with a known chain gap whose ETA is still ahead of us and within the hard cap
        int k = 0; std::vector<double> etas;
        for (auto & p : pred) {
            if (!p.in_flight || p.gap_ema_us <= 0) continue;
            const double since = std::chrono::duration<double, std::micro>(now - p.t_deliver).count();
            const double eta = p.gap_ema_us + 2.0 * p.gap_var_us - since;   // margin: 2x mean abs deviation
            if (eta <= 0 || eta > self->hold_max_us) continue;               // overdue (not coming) or too far away
            etas.push_back(eta); k++;
        }
        if (k == 0) { if (n_expected_out) *n_expected_out = 0; return 0; }
        std::sort(etas.begin(), etas.end());
        // take the largest prefix of arrivals (sorted by ETA) that still pays for its own wait
        int best = 0; double best_wait = 0;
        for (int i = 0; i < k; i++) {
            const double wait = etas[i];
            const double value = (i + 1) * fixed_us * busy_ema * self->hold_gain;
            if (wait < value) { best = i + 1; best_wait = wait; }
        }
        if (n_expected_out) *n_expected_out = best;
        return best > 0 ? (int) best_wait + 1 : 0;
    }

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
                    if (self->policy == "adaptive") {
                        // Hold the queue only for sequences predicted to resubmit within the budget. Each hold is
                        // re-evaluated when a new step lands (cv notify): a sequence that arrived is no longer expected,
                        // one that is now overdue is dropped, and the wait shrinks to the latest remaining ETA.
                        // The budget itself follows GPU pressure (busy_ema): idle GPU -> ~0 hold -> pure latency mode.
                        // The oldest queued step bounds the total hold: however many arrivals keep being predicted,
                        // no step may be held longer than hold_max_us in total. Without this bound a steady stream
                        // of predicted arrivals (N sessions resubmitting in staggered fashion) keeps the worker
                        // waiting forever and the steps already queued never run (seen as two sessions at 5 s
                        // decision latency while the other ten sat at 0.5 s).
                        const auto t_hold0 = std::chrono::steady_clock::now();
                        int n_expected = 0; bool held = false; size_t q0 = queue.size(); int n_iter = 0;
                        for (;;) {
                            if (!running) break;
                            const auto now = std::chrono::steady_clock::now();
                            auto oldest = now; for (auto * s : queue) oldest = std::min(oldest, s->t_enq);
                            const long held_us = std::chrono::duration_cast<std::chrono::microseconds>(now - oldest).count();
                            if (held_us >= self->hold_max_us) break;
                            int wait_us = adaptive_hold_us(now, queue.size(), &n_expected);
                            if (wait_us <= 0) break;
                            wait_us = std::min<int>(wait_us, self->hold_max_us - (int) held_us);
                            held = true; n_iter++;
                            const size_t before = queue.size();
                            cv.wait_for(lk, std::chrono::microseconds(wait_us), [&]{ return !running || queue.size() != before; });
                        }
                        if (held) {
                            const double held_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_hold0).count();
                            if (self->trace) fprintf(stderr, "[hold] q0=%zu q=%zu iters=%d expected=%d held_ms=%.2f busy=%.2f F=%.1f\n", q0, queue.size(), n_iter, n_expected, held_ms, busy_ema, fixed_cost_ema_us / 1000.0);
                            std::lock_guard<std::mutex> sg(stats_mtx);
                            self->hold_ms_total += held_ms; self->n_hold++; if (queue.size() > q0) self->n_hold_hit++;
                        }
                    } else if (ripe) {
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
        const int rc = llama_decode(ctx, b);
        if (rc != 0) fprintf(stderr, "[exec] llama_decode rc=%d (steps=%zu tok=%d)\n", rc, batch.size(), n_tok);
        // llama_decode returns before the device finishes; the logits read below synchronizes anyway, so do it here
        // to make `ms` the real forward time (the number the adaptive hold budget is derived from).
        llama_synchronize(ctx);
        const auto t1 = std::chrono::steady_clock::now();
        self->n_forward += llama_context_graph_compute_count(ctx) - forward_last; forward_last = llama_context_graph_compute_count(ctx);
        for (int j = 0; j < n_dummy; j++) llama_memory_seq_rm(llama_get_memory(ctx), self->n_seq + j, -1, -1);
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // GPU pressure: fraction of wall time the executor spent inside forwards, EMA over batches
        if (t_last_batch_end.time_since_epoch().count() != 0) {
            const double wall = std::chrono::duration<double, std::milli>(t1 - t_last_batch_end).count();
            const double b = wall > 0 ? std::min(1.0, ms / wall) : 1.0;
            busy_ema = busy_ema <= 0 ? b : 0.8 * busy_ema + 0.2 * b;
        }
        t_last_batch_end = t1;
        if (batch.size() == 1 && n_tok == 1) fixed_cost_ema_us = fixed_cost_ema_us <= 0 ? ms * 1000.0 : 0.9 * fixed_cost_ema_us + 0.1 * ms * 1000.0;
        const bool trace = self->trace;
        if (trace) { int mx = 0; for (auto * s : batch) mx = std::max(mx, s->n); fprintf(stderr, "[exec] steps=%zu tok=%d max_n=%d ms=%.1f qwait_avg=%.1f qwait_max=%.1f busy=%.2f\n", batch.size(), n_tok, mx, ms, qwait_sum / batch.size(), qwait_max, busy_ema); }
        {
            std::lock_guard<std::mutex> sg(stats_mtx);
            self->qwait_ms_total += qwait_sum; self->qwait_ms_max = std::max(self->qwait_ms_max, qwait_max);
            self->n_pad_tokens += n_dummy;
            self->decode_ms_total += ms;
            self->n_batches++; self->n_steps += batch.size(); self->n_tokens_total += n_tok;
            for (auto * s : batch) self->step_len_hist[s->n]++;
            self->batch_width_hist[(int) batch.size()]++;
            self->max_steps_in_batch = std::max<uint64_t>(self->max_steps_in_batch, batch.size());
            self->busy_fraction = busy_ema;
            if (fixed_cost_ema_us > 0) self->fixed_cost_ms = fixed_cost_ema_us / 1000.0;
        }
        // scatter outputs. llama_get_logits_ith / get_embeddings_ith take the BATCH TOKEN index (they map it through
        // output_ids to the compacted output row), not the output row number. Walk the batch by token offset and
        // ask for the tokens whose out[] flag was set: all of a step's tokens when it asked for all_outputs, else its last.
        int tok_off = 0;
        for (auto * s : batch) {
            omni_llm_step_result r; r.ok = (rc == 0);
            const int n_out = s->all_outputs ? s->n : 1;
            if (r.ok) {
                r.logits.resize((size_t) n_out * n_vocab);
                r.hidden.resize((size_t) n_out * n_embd);
                for (int j = 0; j < n_out; j++) {
                    const int ti = s->all_outputs ? tok_off + j : tok_off + s->n - 1;
                    const float * lg = llama_get_logits_ith(ctx, ti);
                    const float * hd = llama_get_embeddings_ith(ctx, ti);
                    if (!lg) { r.ok = false; fprintf(stderr, "[exec] no logits for step seq=%d n=%d (batch token %d)\n", s->seq, s->n, ti); break; }
                    memcpy(r.logits.data() + (size_t) j * n_vocab, lg, n_vocab * sizeof(float));
                    if (hd) memcpy(r.hidden.data() + (size_t) j * n_embd, hd, n_embd * sizeof(float));
                }
            }
            tok_off += s->n;
            note_delivered(s, std::chrono::steady_clock::now());
            s->done.set_value(std::move(r));
        }
    }
};

omni_llm_step_result omni_llm_exec::run(int seq, int pos0, const float * embd, int n_tokens, bool all_outputs) {
    auto * s = new impl::step(); s->seq = seq; s->pos0 = pos0; s->embd = embd; s->n = n_tokens; s->all_outputs = all_outputs; s->t_enq = std::chrono::steady_clock::now();
    auto fut = s->done.get_future();
    { std::lock_guard<std::mutex> lk(p->mtx); p->queue.push_back(s); p->note_submitted(s); }
    p->cv.notify_all();
    omni_llm_step_result r = fut.get();
    delete s;
    return r;
}

omni_llm_exec::snapshot_t omni_llm_exec::snapshot() const {
    std::lock_guard<std::mutex> sg(p->stats_mtx);
    snapshot_t s;
    s.n_batches = n_batches; s.n_steps = n_steps; s.n_tokens_total = n_tokens_total; s.n_forward = n_forward;
    s.max_steps_in_batch = max_steps_in_batch; s.n_hold = n_hold; s.n_hold_hit = n_hold_hit;
    s.decode_ms_total = decode_ms_total; s.qwait_ms_total = qwait_ms_total; s.qwait_ms_max = qwait_ms_max;
    s.hold_ms_total = hold_ms_total; s.busy_fraction = busy_fraction; s.fixed_cost_ms = fixed_cost_ms;
    s.batch_width_hist = batch_width_hist;
    return s;
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
    if (const char * v = knob("HOLD_GAIN")) e->hold_gain = atof(v);
    e->hold_max_us         = knob_i("HOLD_MAX_US", 20000);
    e->chain_gap_max_us    = knob_i("CHAIN_GAP_MAX_US", 60000);
    e->fixed_cost_us       = knob_i("FIXED_COST_US", 6500);
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
    e->p->pred.resize(n_seq);
    e->fixed_cost_ms = e->fixed_cost_us / 1000.0;
    if (g_lockstep && (env_prefix == nullptr || env_prefix[0] == 0)) {
        std::lock_guard<std::mutex> g(g_lockstep->m);
        g_lockstep->members.push_back(e->p.get());
        e->p->lockstep_member = true;
    }
    e->p->worker = std::thread([pi = e->p.get()]{ pi->loop(); });
    return e;
}
