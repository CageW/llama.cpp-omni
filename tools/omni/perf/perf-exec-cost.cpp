// Service-time curve of the shared LLM executor.
//
// The adaptive batch scheduler needs to know how the cost of one llama_decode grows with the
// number of sequences in it and with the tokens per sequence. This tool measures exactly that on
// the real model, through the same omni_llm_exec path the server uses: it creates an executor with
// N sequences, submits `width` identical steps of `n_tok` tokens from `width` threads at the same
// instant, and records the wall time of the merged forward. Nothing else runs, so the batch is
// always exactly as wide as asked.
//
//   llama-omni-perf-exec-cost -m llm.gguf --widths 1,2,4,8,12,16,24 --lens 1,4,8,80 [--reps 5]
//
// Output (stdout, CSV): kind,width,n_tok,rep,ms_forward,ms_per_step,n_forward
// where kind is "decode" for n_tok <= 8 and "prefill" otherwise, and n_forward is how many graph
// computes (ubatches) the one logical batch was split into. For prefill that is ceil(width*n_tok/n_ubatch):
// with OMNI_EXEC_UBATCH=128 a 4x80 batch is 3 forwards, so the merged cost tracks total tokens, not width;
// with n_ubatch 512 it is 1 forward up to width 6. The per-step numbers therefore depend on the ubatch
// setting the server runs with, and the controller must be calibrated at that setting.
//
// stderr will show "invalid logits id" lines: they come from llama_get_logits_ith on rows this tool
// never asked an output for (the tool only reads the last token of each step) and are harmless here.
#include "llm-exec.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>

static std::vector<int> parse_list(const std::string & s) {
    std::vector<int> out; size_t p = 0;
    while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) out.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; }
    return out;
}

int main(int argc, char ** argv) {
    std::string llm_path; std::vector<int> widths{1, 2, 4, 8, 16}; std::vector<int> lens{1, 80}; int reps = 5; int n_ctx = 8192; int n_ubatch = 512; int n_seq_max = 0; int warm = 2; bool mixed = false; int verify_w = 0;
    ggml_type ctk = GGML_TYPE_F16;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) llm_path = argv[++i];
        else if (a == "--widths" && i + 1 < argc) widths = parse_list(argv[++i]);
        else if (a == "--lens" && i + 1 < argc) lens = parse_list(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--warm" && i + 1 < argc) warm = atoi(argv[++i]);
        else if (a == "--mixed") mixed = true;
        else if (a == "--verify" && i + 1 < argc) verify_w = atoi(argv[++i]);
        else if (a == "-c" && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (a == "--ub" && i + 1 < argc) n_ubatch = atoi(argv[++i]);
        else if (a == "--nseq" && i + 1 < argc) n_seq_max = atoi(argv[++i]);
        else if (a == "--ctk" && i + 1 < argc) { std::string t = argv[++i]; ctk = t == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; }
        else { fprintf(stderr, "usage: %s -m llm.gguf [--widths a,b,..] [--lens a,b,..] [--reps k] [-c n_ctx] [--ub n] [--ctk f16|q8_0]\n", argv[0]); return 1; }
    }
    if (llm_path.empty()) { fprintf(stderr, "need -m\n"); return 1; }
    if (n_seq_max <= 0) for (int w : widths) n_seq_max = std::max(n_seq_max, w);
    if (verify_w > 0) n_seq_max = std::max(n_seq_max, verify_w + 1);

    llama_backend_init();
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(llm_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 4096; cp.n_ubatch = n_ubatch;
    cp.type_k = ctk; cp.type_v = ctk;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads = 4; cp.n_threads_batch = 4;

    // The executor's default window breaks as soon as any decode-length step is queued, so a width-W
    // decode batch would be split at W=1. The "ripe" policy with a large prefill target and a long deadline
    // holds prefills until `target` are queued; for decode-length steps we use PREFILL gather via a low
    // long_threshold (0) so every step counts as "long" and is held until `width` have arrived.
    setenv("OMNI_EXEC_POLICY", "ripe", 0);
    setenv("OMNI_EXEC_LONG_THRESHOLD", "0", 0);
    setenv("OMNI_EXEC_PREFILL_DEADLINE_US", "2000000", 0);   // 2 s: only fires if a thread is late
    setenv("OMNI_EXEC_WINDOW_US", "0", 0);
    auto ex = omni_llm_exec_create(model, cp, n_seq_max, n_ctx);
    if (!ex) { fprintf(stderr, "exec create failed\n"); return 1; }
    const int n_embd = ex->n_embd;
    std::vector<int> seqs(n_seq_max);
    for (int i = 0; i < n_seq_max; ++i) seqs[i] = ex->alloc_seq();

    int max_len = 0; for (int l : lens) max_len = std::max(max_len, l);
    std::vector<float> embd((size_t) max_len * n_embd, 0.01f);

    if (verify_w > 0) {
        // ---- correctness of merged batches ----
        // Reference: sequence 0 alone runs a 12-token prefill then 3 single-token decode steps, each as its own
        // single-step batch (the path every earlier single-session result was produced with). Then W other
        // sequences run the SAME inputs but with their steps merged into one forward per round (decode-wide),
        // and in a mixed shape (half the sequences prefill while the other half decode in the same forward).
        // The logits of every sequence must match the reference up to F16 noise; a wrong-row / wrong-sequence
        // scatter shows up as a large max-abs-diff and different argmax. Nothing in the server checks this.
        const int W = verify_w, L = 12, n_vocab = ex->n_vocab;
        std::vector<float> pre((size_t) L * n_embd), dec((size_t) 3 * n_embd);
        { uint32_t st = 12345u; auto rnd = [&]{ st = st * 1664525u + 1013904223u; return ((st >> 8) & 0xffff) / 65535.0f - 0.5f; };
          for (auto & v : pre) v = rnd() * 0.2f; for (auto & v : dec) v = rnd() * 0.2f; }
        auto argmax = [&](const std::vector<float> & lg) { return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin()); };
        auto maxdiff = [&](const std::vector<float> & a, const std::vector<float> & b) { double m = 0; for (size_t i = 0; i < a.size() && i < b.size(); i++) m = std::max(m, (double) std::fabs(a[i] - b[i])); return m; };
        // one round of `total` simultaneous steps; steps[i] = {seq, pos0, embd, n}
        struct vstep { int seq, pos0, n; const float * embd; };
        auto round = [&](const std::vector<vstep> & steps) {
            ex->prefill_target = (int) steps.size();
            std::vector<omni_llm_step_result> res(steps.size());
            std::mutex m; std::condition_variable cv; bool go = false; std::atomic<int> ready{0};
            std::vector<std::thread> th;
            const uint64_t nb0 = ex->n_batches;
            for (size_t i = 0; i < steps.size(); ++i) th.emplace_back([&, i]() {
                { std::unique_lock<std::mutex> lk(m); ready++; cv.wait(lk, [&]{ return go; }); }
                res[i] = ex->run(steps[i].seq, steps[i].pos0, steps[i].embd, steps[i].n, false);
            });
            while (ready.load() < (int) steps.size()) std::this_thread::yield();
            { std::lock_guard<std::mutex> lk(m); go = true; } cv.notify_all();
            for (auto & t : th) t.join();
            if (ex->n_batches - nb0 != 1) fprintf(stderr, "[verify] round of %zu steps ran as %llu batches (wanted 1)\n", steps.size(), (unsigned long long) (ex->n_batches - nb0));
            return res;
        };
        for (int i = 0; i < n_seq_max; ++i) ex->seq_clear(seqs[i]);
        // reference on seq 0
        std::vector<std::vector<float>> ref(4);
        ref[0] = round({{seqs[0], 0, L, pre.data()}})[0].logits;
        for (int k = 0; k < 3; k++) ref[k + 1] = round({{seqs[0], L + k, 1, dec.data() + (size_t) k * n_embd}})[0].logits;
        printf("verify: reference argmax prefill=%d dec0=%d dec1=%d dec2=%d (ubatch=%d, unified=%s)\n", argmax(ref[0]), argmax(ref[1]), argmax(ref[2]), argmax(ref[3]), n_ubatch, getenv("OMNI_EXEC_UBATCH") ? "?" : "?");
        // A) prefill-wide: W sequences prefill in one forward
        {
            std::vector<vstep> st; for (int i = 1; i <= W; i++) st.push_back({seqs[i], 0, L, pre.data()});
            auto r = round(st);
            double md = 0; int bad = 0;
            for (int i = 0; i < W; i++) { md = std::max(md, maxdiff(r[i].logits, ref[0])); bad += argmax(r[i].logits) != argmax(ref[0]); }
            printf("verify,prefill_wide,W=%d,maxabsdiff=%.4g,argmax_mismatch=%d/%d\n", W, md, bad, W);
        }
        // B) decode-wide: 3 rounds, all W sequences submit their next token together
        for (int k = 0; k < 3; k++) {
            std::vector<vstep> st; for (int i = 1; i <= W; i++) st.push_back({seqs[i], L + k, 1, dec.data() + (size_t) k * n_embd});
            auto r = round(st);
            double md = 0; int bad = 0;
            for (int i = 0; i < W; i++) { md = std::max(md, maxdiff(r[i].logits, ref[k + 1])); bad += argmax(r[i].logits) != argmax(ref[k + 1]); }
            printf("verify,decode_wide_round%d,W=%d,maxabsdiff=%.4g,argmax_mismatch=%d/%d\n", k, W, md, bad, W);
        }
        // C) mixed: reset; first half prefill while second half (already prefilled + decoded to L+? no: re-prefilled alone) decode step 0
        for (int i = 1; i <= W; i++) ex->seq_clear(seqs[i]);
        const int h = std::max(1, W / 2);
        for (int i = h + 1; i <= W; i++) round({{seqs[i], 0, L, pre.data()}});   // second half: prefill alone (known-good path)
        {
            std::vector<vstep> st;
            for (int i = 1; i <= h; i++) st.push_back({seqs[i], 0, L, pre.data()});
            for (int i = h + 1; i <= W; i++) st.push_back({seqs[i], L, 1, dec.data()});
            auto r = round(st);
            double mdp = 0, mdd = 0; int badp = 0, badd = 0;
            for (int i = 0; i < h; i++) { mdp = std::max(mdp, maxdiff(r[i].logits, ref[0])); badp += argmax(r[i].logits) != argmax(ref[0]); }
            for (int i = h; i < W; i++) { mdd = std::max(mdd, maxdiff(r[i].logits, ref[1])); badd += argmax(r[i].logits) != argmax(ref[1]); }
            printf("verify,mixed,W=%d,prefill_half:maxabsdiff=%.4g,argmax_mismatch=%d/%d;decode_half:maxabsdiff=%.4g,argmax_mismatch=%d/%d\n", W, mdp, badp, h, mdd, badd, W - h);
        }
        // D) decode-wide at DIFFERENT positions per sequence (the real server shape): seq i has i extra tokens of history
        for (int i = 1; i <= W; i++) ex->seq_clear(seqs[i]);
        std::vector<int> pos(W + 1, L);
        for (int i = 1; i <= W; i++) { round({{seqs[i], 0, L, pre.data()}}); for (int e = 0; e < i % 3; e++) { round({{seqs[i], pos[i], 1, dec.data() + (size_t) e * n_embd}}); pos[i]++; } }
        {
            // each seq now sits at L + (i%3); the next token it submits is dec[(i%3)], whose reference is ref[(i%3)+1]
            std::vector<vstep> st; for (int i = 1; i <= W; i++) st.push_back({seqs[i], pos[i], 1, dec.data() + (size_t) (i % 3) * n_embd});
            auto r = round(st);
            double md = 0; int bad = 0;
            for (int i = 1; i <= W; i++) { md = std::max(md, maxdiff(r[i - 1].logits, ref[(i % 3) + 1])); bad += argmax(r[i - 1].logits) != argmax(ref[(i % 3) + 1]); }
            printf("verify,decode_wide_staggered_pos,W=%d,maxabsdiff=%.4g,argmax_mismatch=%d/%d\n", W, md, bad, W);
        }
        fflush(stdout);
        if (widths.empty()) { ex.reset(); llama_model_free(model); llama_backend_free(); return 0; }
    }
    printf("kind,width,n_tok,rep,ms_forward,ms_per_step,n_forward\n");
    // measure one (kind, W, L, n_extra_decode) point: W steps of L tokens (+ n_extra 1-token steps on other seqs)
    auto measure = [&](const char * kind, int W, int L, int n_extra) {
        const int total = W + n_extra;
        if (total > n_seq_max) return;
        ex->prefill_target = total;   // ripe with LONG_THRESHOLD=0: hold until exactly `total` steps queued, then one merged forward
        for (int rep = -warm; rep < reps; ++rep) {
            for (int i = 0; i < total; ++i) ex->seq_clear(seqs[i]);
            std::mutex m; std::condition_variable cv; bool go = false; std::atomic<int> ready{0};
            std::vector<std::thread> th;
            std::vector<double> t_done(total);
            const uint64_t fwd0 = ex->n_forward;
            const uint64_t nb0 = ex->n_batches;
            auto t_start_holder = std::chrono::steady_clock::now();
            for (int i = 0; i < total; ++i) {
                const int n = i < W ? L : 1;
                th.emplace_back([&, i, n]() {
                    { std::unique_lock<std::mutex> lk(m); ready++; cv.wait(lk, [&]{ return go; }); }
                    auto r = ex->run(seqs[i], 0, embd.data(), n, false);
                    if (!r.ok) fprintf(stderr, "step failed w=%d l=%d\n", W, n);
                    t_done[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start_holder).count();
                });
            }
            while (ready.load() < total) std::this_thread::yield();
            { std::lock_guard<std::mutex> lk(m); go = true; t_start_holder = std::chrono::steady_clock::now(); }
            cv.notify_all();
            for (auto & t : th) t.join();
            const double ms = *std::max_element(t_done.begin(), t_done.end());
            const uint64_t nfwd = ex->n_forward - fwd0;
            const uint64_t nb = ex->n_batches - nb0;
            if (nb != 1) fprintf(stderr, "[warn] %s w=%d l=%d rep=%d: %llu batches (wanted 1) -> measurement is not one merged forward\n", kind, W, L, rep, (unsigned long long) nb);
            if (rep >= 0) {
                printf("%s,%d,%d,%d,%.2f,%.3f,%llu\n", kind, W, L, rep, ms, ms / W, (unsigned long long) nfwd);
                fflush(stdout);
            }
        }
    };
    for (int L : lens) {
        for (int W : widths) {
            measure(L <= 8 ? "decode" : "prefill", W, L, 0);
            // mixed: the same W prefills plus W single-token decodes in ONE forward (what a policy that does not
            // separate the two classes produces). Tells whether mixing costs GPU time or "only" decode latency.
            if (mixed && L > 8) measure("mixed", W, L, W);
        }
    }
    ex.reset();
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
