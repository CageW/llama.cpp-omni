#include "ws_handler.h"

#undef NDEBUG
#include <cassert>
#include <functional>
#include <mutex>

static int release_count = 0;

static void record_release(omni_context *) {
    ++release_count;
}

int main() {
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
    return 0;
}
