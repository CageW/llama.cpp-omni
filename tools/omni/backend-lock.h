#pragma once
// Per-backend compute lock for weights shared across sessions.
//
// ggml backends keep a per-instance memory pool (CUDA: a LIFO VMM pool). Two threads
// running graphs on the SAME backend instance interleave pool alloc/free and trip
// GGML_ASSERT(ptr == pool_addr + pool_used). Session clones share the Token2Wav,
// projector and audition backends, so every compute on those goes through this lock.
#include "ggml-backend.h"
#include <map>
#include <mutex>
#include <memory>

inline std::mutex & omni_backend_mutex(ggml_backend_t backend) {
    static std::mutex map_mtx;
    static std::map<ggml_backend_t, std::unique_ptr<std::mutex>> mutexes;
    std::lock_guard<std::mutex> g(map_mtx);
    auto it = mutexes.find(backend);
    if (it == mutexes.end()) {
        it = mutexes.emplace(backend, std::make_unique<std::mutex>()).first;
    }
    return *it->second;
}
