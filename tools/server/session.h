#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct omni_context;

enum class SessionMode {
    FULL_DUPLEX,
    TURN_BASED,
};

enum class SessionState {
    UNINITIALIZED,
    ACTIVE,
    CLOSED,
};

struct OmniSession {
    std::string session_id;
    SessionState state = SessionState::UNINITIALIZED;
    SessionMode mode = SessionMode::FULL_DUPLEX;
    omni_context * octx = nullptr;
    bool owns_octx = false;
    double created_at = 0.0;
    std::function<void()> close_ws;
};

// ============================================================================
// OmniSlot — a pool entry that holds one pre-cloned omni_context.
//
// The server creates N slots at startup (one root + N-1 clones), each wrapping
// a ready-to-use omni_context. A slot transitions:
//
//   IDLE → ACTIVE  : checkout() — caller calls reset+begin on the octx
//   ACTIVE → IDLE  : release() — slot is returned to the pool
//
// The octx pointer is never null after pool construction and never freed during
// normal operation (only on shutdown or if the slot itself is destroyed).
// ============================================================================

enum class SlotState { IDLE, ACTIVE };

struct OmniSlot {
    int          slot_idx   = 0;
    SlotState    state      = SlotState::IDLE;
    omni_context * octx     = nullptr;   // non-owning after pool init; root owns on shutdown
    bool         owns_octx  = false;     // true only for the root slot
};

// ============================================================================
// SessionPool — N-slot pool for multi-session omni serving.
//
// Usage:
//   1. Construct with capacity 0; call init_slots() once all omni_context*
//      pointers are ready (after omni_init + omni_session_clone calls).
//   2. checkout_slot()  → waits for a free slot; returns slot index + pointer.
//   3. return_slot(idx) → marks slot idle, wakes waiters.
//   4. shutdown()       → closes all slots, stops waiters, frees root context.
//
// Thread-safe. All public methods may be called from any thread.
// ============================================================================

class SessionPool {
public:
    SessionPool() = default;
    ~SessionPool() { shutdown(); }

    // Initialize the pool from pre-built contexts.
    // `contexts[0]` is the root (owns the model weights); the rest are clones.
    // May only be called once.
    void init_slots(std::vector<omni_context *> contexts);

    // Block until a slot becomes available, then mark it ACTIVE and return
    // a pointer to it. Returns nullptr if the pool has been shut down.
    OmniSlot * checkout_slot();

    // Mark the slot at `idx` as IDLE and wake one waiter.
    void return_slot(int idx);

    // Return the omni_context* for an active slot (for /sessions/:id/close).
    // Returns nullptr if not found.
    omni_context * get_octx(int slot_idx);

    // Return the number of configured slots (0 = single-session legacy mode).
    int capacity() const { return static_cast<int>(slots_.size()); }

    // Stop accepting new checkouts. Ongoing sessions drain naturally.
    void shutdown();

private:
    mutable std::mutex          mtx_;
    std::condition_variable     cv_;
    std::vector<OmniSlot>       slots_;
    bool                        stopped_ = false;
};

// ============================================================================
// SessionManager — maps active WebSocket sessions to pool slots.
//
// When the pool is in use (capacity > 0), each WebSocket /backend connection
// checks out a slot and registers a session here. The /sessions/:id/close
// endpoint looks up the session to find the associated slot and octx.
//
// Legacy single-session mode (pool capacity == 0) is preserved: allocate()
// still enforces at-most-one and falls back to the shared_octx path in
// ws_handler.
// ============================================================================

class SessionManager {
public:
    ~SessionManager();

    // ---- slot-pool mode ----

    // Attach a pool so the manager can route session IDs → slots.
    void set_pool(SessionPool * pool);

    // Register a session that was allocated from slot `slot_idx`.
    // Returns the new session_id.
    std::string register_slot_session(int slot_idx, omni_context * octx);

    // Look up a session. Returns nullptr if not found.
    OmniSession * get(const std::string & session_id);

    // Register/trigger transport close.
    void set_close_callback(const std::string & session_id, std::function<void()> cb);
    void request_transport_close(const std::string & session_id);

    // Unregister the session and return its slot to the pool.
    void close(const std::string & session_id);

    // Called on WS disconnect — auto-close whatever session this was.
    // Returns the slot_idx that was released (-1 if none).
    int on_disconnect(const std::string & session_id);

    // ---- legacy single-session mode ----

    // Allocate a new session (UNINITIALIZED). Returns session_id, or "" if busy.
    std::string allocate();

    // Activate a legacy session.
    bool activate(const std::string & session_id, omni_context * octx, bool owns_octx);

    // True if any session is currently ACTIVE (legacy path).
    bool has_active() const;

    // Close all sessions (shutdown cleanup).
    void shutdown();

private:
    mutable std::mutex mtx_;

    // Pool-mode sessions (session_id → session).
    // Multiple concurrent sessions are allowed in pool mode.
    std::unordered_map<std::string, std::unique_ptr<OmniSession>> pool_sessions_;

    // Legacy single-session slot.
    std::unique_ptr<OmniSession> active_;

    // Back-reference to pool so close() can return_slot().
    SessionPool * pool_ = nullptr;
    // Maps session_id → slot_idx for pool-mode sessions.
    std::unordered_map<std::string, int> session_slot_;

    std::string generate_uuid();
};
