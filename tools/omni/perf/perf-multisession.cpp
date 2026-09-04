// Multi-session duplex capacity probe.
//
// Loads ONE omni_context (all weights), clones N-1 session contexts that share those weights
// (omni_session_clone), then drives all N sessions concurrently with the same duplex test
// set at a fixed frame interval. Reports per-frame LLM decision latency (p50/p95/max per
// session and pooled), the number of speak/listen frames, GPU memory before/after clone
// and at the end, and whether every session stayed under the frame interval.
#include "omni.h"
#include "common.h"
#include "arg.h"
#include "llama.h"
#include "log.h"
#include "ggml-backend.h"
#include "token2wav/token2wav.h"
#include "vision.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <numeric>
#include <cmath>

static bool file_exists(const std::string & p) { FILE * f = fopen(p.c_str(), "rb"); if (f) { fclose(f); return true; } return false; }
static std::string parent_dir(const std::string & p) { size_t k = p.find_last_of("/\\"); return k == std::string::npos ? "." : p.substr(0, k); }

static size_t gpu_used_mib() {
    size_t used = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        size_t fr = 0, tot = 0; ggml_backend_dev_memory(dev, &fr, &tot);
        used += (tot - fr) / (1024 * 1024);
    }
    return used;
}

struct SessStats {
    std::vector<double> ms_total, ms_submit, ms_decode;
    int speak = 0, listen = 0, fail = 0;
    double first_audio_ms = -1;
    int wav_chunks = 0;
};

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t) std::min<double>(v.size() - 1, std::floor(p / 100.0 * (v.size() - 1) + 0.5));
    return v[i];
}

int main(int argc, char ** argv) {
    std::string llm_path, prefix; int cnt = 36, n_sessions = 4, interval_ms = 1000, n_ctx = 8192; bool use_tts = true;
    std::string out_csv; ggml_type ctk = GGML_TYPE_F16; int n_ubatch = 512; int n_threads = 4; std::vector<std::string> t2w_devs{"gpu:0"}; bool use_exec = false; std::vector<std::string> enc_devs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) llm_path = argv[++i];
        else if (a == "--test" && i + 2 < argc) { prefix = argv[++i]; cnt = atoi(argv[++i]); }
        else if (a == "-n" && i + 1 < argc) n_sessions = atoi(argv[++i]);
        else if (a == "--interval" && i + 1 < argc) interval_ms = atoi(argv[++i]);
        else if (a == "-c" && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (a == "--no-tts") use_tts = false;
        else if (a == "--csv" && i + 1 < argc) out_csv = argv[++i];
        else if (a == "--ctk" && i + 1 < argc) { std::string t = argv[++i]; ctk = t == "q8_0" ? GGML_TYPE_Q8_0 : t == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16; }
        else if (a == "--ub" && i + 1 < argc) n_ubatch = atoi(argv[++i]);
        else if (a == "--exec") use_exec = true;
        else if (a == "-t" && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (a == "--enc-devices" && i + 1 < argc) { std::string l = argv[++i]; size_t q0 = 0; while (q0 <= l.size()) { size_t q = l.find(',', q0); if (q == std::string::npos) q = l.size(); if (q > q0) enc_devs.push_back(l.substr(q0, q - q0)); q0 = q + 1; } }
        else if (a == "--t2w-devices" && i + 1 < argc) { t2w_devs.clear(); std::string l = argv[++i]; size_t p = 0; while (p <= l.size()) { size_t q = l.find(',', p); if (q == std::string::npos) q = l.size(); if (q > p) t2w_devs.push_back(l.substr(p, q - p)); p = q + 1; } }
        else { fprintf(stderr, "usage: %s -m llm.gguf --test <prefix> <n> [-n sessions] [--interval ms] [-c n_ctx] [--no-tts] [--csv out]\n", argv[0]); return 1; }
    }
    if (llm_path.empty() || prefix.empty()) { fprintf(stderr, "need -m and --test\n"); return 1; }

    common_params params;
    params.model.path = llm_path;
    params.n_ctx = n_ctx; params.n_batch = 2048; params.n_ubatch = n_ubatch; params.n_gpu_layers = 99;
    params.cache_type_k = ctk; params.cache_type_v = ctk; params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.n_parallel = 1;
    params.cpuparams.n_threads = n_threads; params.cpuparams_batch.n_threads = n_threads;   // per-session CPU threads: default -1 = all cores -> N sessions oversubscribe the host
    params.split_mode = LLAMA_SPLIT_MODE_NONE; params.main_gpu = 0;   // LLM/APM/VPM stay on the first visible GPU
    params.sampling.temp = 0.7f; params.sampling.top_p = 0.8f; params.sampling.top_k = 20; params.sampling.penalty_repeat = 1.05f;
    const std::string base = parent_dir(llm_path);
    const bool o5 = llm_path.find("o5") != std::string::npos;
    const std::string tag = o5 ? "MiniCPM-o5" : "MiniCPM-o-4_5";
    params.vpm_model = base + "/vision/" + tag + "-vision-F16.gguf";
    params.apm_model = base + "/audio/" + tag + "-audio-F16.gguf";
    params.tts_model = base + "/tts/" + tag + "-tts-F16.gguf";
    params.tts_bin_dir = base + "/tts";

    llama_backend_init();
    ggml_backend_load_all();
    const size_t mem0 = gpu_used_mib();
    omni_context * root = omni_init(&params, /*media_type*/2, use_tts, params.tts_bin_dir, 99, t2w_devs[0], /*duplex*/true,
                                    nullptr, nullptr, "./tools/omni/output/ms0");
    if (!root) { fprintf(stderr, "omni_init failed\n"); return 1; }
    root->clean_kvcache = true;
    if (use_exec && !omni_enable_llm_exec(root, n_sessions, n_ctx)) { fprintf(stderr, "omni_enable_llm_exec failed\n"); return 1; }
    if (!enc_devs.empty() && !omni_add_encoder_devices(root, enc_devs)) { fprintf(stderr, "omni_add_encoder_devices failed\n"); return 1; }
    const size_t mem_root = gpu_used_mib();

    std::vector<omni_context *> sess{root};
    std::vector<size_t> mem_after_clone{mem_root};
    // extra Token2Wav weight owners on additional GPUs (stage placement experiment)
    std::vector<std::unique_ptr<omni::flow::Token2WavSession>> t2w_owners;
    for (size_t d = 1; d < t2w_devs.size(); ++d) {
        auto o = omni_load_token2wav_owner(root->token2wav_model_dir, t2w_devs[d]);
        if (!o) { fprintf(stderr, "t2w owner on %s failed\n", t2w_devs[d].c_str()); return 1; }
        t2w_owners.push_back(std::move(o));
    }
    printf("[t2w] devices: "); for (auto & d : t2w_devs) printf("%s ", d.c_str()); printf("\n");
    for (int i = 1; i < n_sessions; ++i) {
        omni::flow::Token2WavSession * owner = nullptr;
        if (!t2w_owners.empty()) { size_t k = (size_t) i % t2w_devs.size(); owner = k == 0 ? nullptr : t2w_owners[k - 1].get(); }
        omni_context * c = omni_session_clone(root, &params, "./tools/omni/output/ms" + std::to_string(i), owner);
        if (!c) { fprintf(stderr, "clone %d failed\n", i); return 1; }
        sess.push_back(c);
        mem_after_clone.push_back(gpu_used_mib());
    }
    printf("[mem] baseline %zu MiB | root(all weights + 1 session) %zu MiB (+%zu) | after %d sessions %zu MiB (+%.1f MiB/session)\n",
           mem0, mem_root, mem_root - mem0, n_sessions, mem_after_clone.back(),
           n_sessions > 1 ? (double)(mem_after_clone.back() - mem_root) / (n_sessions - 1) : 0.0);

    // frames
    std::vector<OmniDuplexFrame> frames(cnt);
    for (int il = 0; il < cnt; ++il) {
        char idx[16]; snprintf(idx, sizeof(idx), "%04d", il);
        frames[il].aud_fname = prefix + idx + ".wav";
        std::string img = prefix + idx + ".jpg"; if (file_exists(img)) frames[il].img_fname = img;
        frames[il].user_seq = il + 1;
        if (!file_exists(frames[il].aud_fname)) { fprintf(stderr, "missing %s\n", frames[il].aud_fname.c_str()); return 1; }
    }

    std::vector<SessStats> stats(n_sessions);
    std::vector<std::chrono::high_resolution_clock::time_point> t0(n_sessions);
    for (int i = 0; i < n_sessions; ++i) {
        int si = i;
        sess[i]->audio_output_cb = [&stats, si, &t0](const float *, int, int, bool) {
            auto now = std::chrono::high_resolution_clock::now();
            if (stats[si].first_audio_ms < 0) stats[si].first_audio_ms = std::chrono::duration<double, std::milli>(now - t0[si]).count();
            stats[si].wav_chunks++;
        };
    }
    // begin all sessions (system prompt prefill) sequentially
    for (int i = 0; i < n_sessions; ++i) {
        if (!omni_duplex_session_begin(sess[i], "", "./tools/omni/output/ms" + std::to_string(i))) { fprintf(stderr, "session_begin %d failed\n", i); return 1; }
    }
    const size_t mem_begun = gpu_used_mib();
    printf("[mem] after session_begin x%d: %zu MiB\n", n_sessions, mem_begun);

    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> producers, consumers;
    for (int i = 0; i < n_sessions; ++i) {
        t0[i] = t_start;
        producers.emplace_back([&, i]() {
            for (int il = 0; il < cnt; ++il) {
                std::this_thread::sleep_until(t_start + std::chrono::milliseconds((int64_t) il * interval_ms));
                if (omni_duplex_push_frame(sess[i], frames[il]) < 0) { fprintf(stderr, "[s%d] push %d failed\n", i, il); break; }
            }
        });
        consumers.emplace_back([&, i]() {
            for (int il = 0; il < cnt; ++il) {
                OmniDuplexFrameResult r;
                if (!omni_duplex_wait_next_frame(sess[i], &r, 60000) || !r.ok) { stats[i].fail++; continue; }
                stats[i].ms_total.push_back(r.ms_total);
                stats[i].ms_submit.push_back(r.ms_prefill_submit);
                stats[i].ms_decode.push_back(r.ms_decode);
                (r.is_speak ? stats[i].speak : stats[i].listen)++;
            }
        });
    }
    for (auto & t : producers) t.join();
    for (auto & t : consumers) t.join();
    const size_t mem_peak = gpu_used_mib();
    for (int i = 0; i < n_sessions; ++i) omni_duplex_session_end(sess[i]);

    std::vector<double> pooled, pooled_submit, pooled_decode;
    printf("\n=== %d sessions | interval %d ms | %d frames each | tts=%d ===\n", n_sessions, interval_ms, cnt, (int) use_tts);
    printf("%-4s %8s %8s %8s %6s %6s %5s %10s %5s\n", "sess", "p50", "p95", "max", "speak", "listen", "fail", "1st_audio", "wavs");
    int over = 0;
    for (int i = 0; i < n_sessions; ++i) {
        auto & st = stats[i];
        pooled.insert(pooled.end(), st.ms_total.begin(), st.ms_total.end());
        pooled_submit.insert(pooled_submit.end(), st.ms_submit.begin(), st.ms_submit.end());
        pooled_decode.insert(pooled_decode.end(), st.ms_decode.begin(), st.ms_decode.end());
        for (double v : st.ms_total) if (v > interval_ms) over++;
        printf("s%-3d %8.1f %8.1f %8.1f %6d %6d %5d %10.0f %5d\n", i, pct(st.ms_total, 50), pct(st.ms_total, 95),
               st.ms_total.empty() ? 0 : *std::max_element(st.ms_total.begin(), st.ms_total.end()),
               st.speak, st.listen, st.fail, st.first_audio_ms, st.wav_chunks);
    }
    printf("pooled: p50 %.1f p95 %.1f max %.1f | frames over interval: %d / %zu | gpu peak %zu MiB (per-session %.1f MiB incl. compute)\n",
           pct(pooled, 50), pct(pooled, 95), pooled.empty() ? 0 : *std::max_element(pooled.begin(), pooled.end()),
           over, pooled.size(), mem_peak, n_sessions > 1 ? (double)(mem_peak - mem_root) / (n_sessions - 1) : 0.0);
    printf("components: push->prefill_done p50 %.0f p95 %.0f | decode(stream_decode) p50 %.0f p95 %.0f | rest(queueing) p50 %.0f\n",
           pct(pooled_submit, 50), pct(pooled_submit, 95), pct(pooled_decode, 50), pct(pooled_decode, 95),
           pct(pooled, 50) - pct(pooled_submit, 50) - pct(pooled_decode, 50));
    if (!out_csv.empty()) {
        FILE * f = fopen(out_csv.c_str(), "a");
        if (f) {
            fprintf(f, "%d,%d,%d,%d,%zu,%zu,%zu,%zu,%.1f,%.1f,%.1f,%d,%zu\n", n_sessions, interval_ms, cnt, (int) use_tts,
                    mem0, mem_root, mem_begun, mem_peak, pct(pooled, 50), pct(pooled, 95),
                    pooled.empty() ? 0 : *std::max_element(pooled.begin(), pooled.end()), over, pooled.size());
            fclose(f);
        }
    }
    if (root->llm_exec) printf("[exec] batches %llu steps %llu tokens %llu | avg steps/batch %.2f max %llu | decode wall %.1f s (avg %.1f ms/batch)\n",
        (unsigned long long) root->llm_exec->n_batches, (unsigned long long) root->llm_exec->n_steps, (unsigned long long) root->llm_exec->n_tokens_total,
        root->llm_exec->n_batches ? (double) root->llm_exec->n_steps / root->llm_exec->n_batches : 0.0, (unsigned long long) root->llm_exec->max_steps_in_batch,
        root->llm_exec->decode_ms_total / 1000.0, root->llm_exec->n_batches ? root->llm_exec->decode_ms_total / root->llm_exec->n_batches : 0.0);
    if (root->llm_exec) printf("[exec] forwards %llu (%.1f per batch | %.2f per step) | %.1f ms/forward\n",
        (unsigned long long) root->llm_exec->n_forward,
        root->llm_exec->n_batches ? (double) root->llm_exec->n_forward / root->llm_exec->n_batches : 0.0,
        root->llm_exec->n_steps ? (double) root->llm_exec->n_forward / root->llm_exec->n_steps : 0.0,
        root->llm_exec->n_forward ? root->llm_exec->decode_ms_total / root->llm_exec->n_forward : 0.0);
    if (root->llm_exec) printf("[exec] padding tokens %llu (%.1f%% of real)\n", (unsigned long long) root->llm_exec->n_pad_tokens, root->llm_exec->n_tokens_total ? 100.0 * root->llm_exec->n_pad_tokens / root->llm_exec->n_tokens_total : 0.0);
    if (root->llm_exec) printf("[exec] queue wait per step: avg %.1f ms, max %.1f ms\n", root->llm_exec->n_steps ? root->llm_exec->qwait_ms_total / root->llm_exec->n_steps : 0.0, root->llm_exec->qwait_ms_max);
    if (root->ctx_vision) { uint64_t c = 0, im = 0; vision_batcher_stats(root->ctx_vision, &c, &im); if (c) printf("[vision] batched calls %llu images %llu (avg %.2f/call)\n", (unsigned long long) c, (unsigned long long) im, (double) im / c); }
    if (root->llm_exec) { printf("[exec] step length histogram:"); for (auto & kv : root->llm_exec->step_len_hist) printf(" %d:%llu", kv.first, (unsigned long long) kv.second); printf("\n"); }
    for (int i = n_sessions - 1; i >= 1; --i) omni_free(sess[i]);
    omni_free(root);
    return 0;
}
