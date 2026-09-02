#include "ws_handler.h"
#include "session.h"

#undef NDEBUG
#include <cassert>
#include <functional>
#include <mutex>
#include <string>

static int release_count = 0;
static std::mutex * release_mutex = nullptr;

static void record_release(omni_context *) {
    ++release_count;
}

static void record_release_after_lock(omni_context *) {
    if (release_mutex != nullptr) {
        std::lock_guard<std::mutex> lock(*release_mutex);
    }
    ++release_count;
}

static void test_session_handle_survives_manager_close() {
    SessionManager manager;
    const std::string session_id = manager.allocate();
    assert(!session_id.empty());
    assert(manager.activate(
        session_id, reinterpret_cast<omni_context *>(0x1), false));

    const std::shared_ptr<OmniSession> retained = manager.get(session_id);
    assert(retained);
    assert(manager.is_active(session_id));

    manager.close(session_id);
    assert(retained->state == SessionState::CLOSED);
    assert(!manager.is_active(session_id));
}

static void test_full_duplex_speak_completion_boundary() {
    // Full-duplex audio is asynchronous and a SPEAK turn may contain several
    // input responses. Flush their response.done events at the following
    // LISTEN boundary, after all audio for the turn has drained.
    assert(should_flush_full_duplex_speak_responses(true));
    assert(!should_flush_full_duplex_speak_responses(false));
}

int main() {
    assert(!should_remove_prefill_audio(
        "/tmp/voice.wav", "/tmp/voice.wav", true));
    assert(should_remove_prefill_audio(
        "/tmp/voice.wav", "/tmp/tts-voice.wav", true));
    assert(should_remove_prefill_audio(
        "/tmp/voice.wav", "/tmp/voice.wav", false));
    assert(!should_remove_prefill_audio("", "", false));
    assert(!should_materialize_tts_ref_audio(false, "encoded-audio"));
    assert(should_materialize_tts_ref_audio(true, "encoded-audio"));
    assert(!should_materialize_tts_ref_audio(true, ""));

    std::mutex octx_mutex;
    omni_context * shared_octx = nullptr;
    omni_context * pending_octx = reinterpret_cast<omni_context *>(0x1);
    omni_context * other_octx = reinterpret_cast<omni_context *>(0x2);

    assert(!release_unactivated_octx_if_owned(
        pending_octx, shared_octx, octx_mutex, false, record_release));

    shared_octx = pending_octx;
    assert(release_unactivated_octx_if_owned(
        pending_octx, shared_octx, octx_mutex, false, record_release));
    assert(shared_octx == nullptr);
    assert(release_count == 1);

    shared_octx = pending_octx;
    assert(!release_unactivated_octx_if_owned(
        other_octx, shared_octx, octx_mutex, false, record_release));
    assert(shared_octx == pending_octx);
    assert(release_count == 1);

    assert(!release_unactivated_octx_if_owned(
        pending_octx, shared_octx, octx_mutex, true, record_release));
    assert(shared_octx == pending_octx);
    assert(release_count == 1);

    shared_octx = pending_octx;
    assert(release_failed_shared_octx_if_owned(
        pending_octx, shared_octx, octx_mutex, record_release));
    assert(shared_octx == nullptr);
    assert(release_count == 2);
    shared_octx = pending_octx;
    release_mutex = &octx_mutex;
    assert(release_unactivated_octx_if_owned(
        pending_octx, shared_octx, octx_mutex, false, record_release_after_lock));
    release_mutex = nullptr;
    assert(shared_octx == nullptr);
    assert(release_count == 3);
    test_session_handle_survives_manager_close();
    test_full_duplex_speak_completion_boundary();
    return 0;
}
