#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <mutex>

struct omni_context;
struct common_params;
struct llama_model;
struct llama_context;
class SessionManager;
class SessionPool;

namespace httplib {
namespace ws { class WebSocket; }
}

// ============================================================================
// WS /backend handler — main entry point called from server.cpp
//
// Two operation modes:
//
//  Pool mode  (pool != nullptr):
//    handle_ws_backend blocks on pool->checkout_slot() to obtain an idle
//    omni_context. The context is reset for the new session via
//    omni_prepare_for_reuse() + omni_duplex_session_begin() and returned to
//    the pool via pool->return_slot() when the WS closes. shared_octx and
//    octx_mutex are ignored in this path.
//
//  Legacy single-session mode (pool == nullptr):
//    Behaviour is unchanged from the original implementation: one active
//    session at a time, shared_octx is reused or reallocated per session,
//    and octx_mutex serialises access.
// ============================================================================

void handle_ws_backend(httplib::ws::WebSocket & ws,
                       SessionManager & session_mgr,
                       common_params & params_base,
                       llama_model * model,
                       llama_context * ctx,
                       omni_context *& shared_octx,  // used only in legacy mode
                       std::mutex & octx_mutex,       // used only in legacy mode
                       SessionPool * pool = nullptr); // multi-session pool

// ============================================================================
// Helpers: base64 audio/JPEG → temp files
// ============================================================================

struct TempMediaFiles {
    std::string audio_path;      // WAV file path (empty if no audio)
    std::string image_path;      // PNG/JPEG file path (empty if no image)
    
    // Write base64 float32 PCM to a temp WAV file
    // Returns empty string on failure
    static std::string write_audio_wav(const std::string & b64, const std::string & temp_dir, int counter);
    
    // Write base64 JPEG/PNG bytes to a temp file
    // Returns empty string on failure
    static std::string write_image_jpeg(const std::string & b64, const std::string & temp_dir, int counter);
    
    // Create a temp file from raw bytes
    static std::string write_temp_file(const std::string & temp_dir, const std::string & prefix,
                                       const std::string & suffix, const void * data, size_t len);
    
    // Clean up temp files
    void cleanup();
};
