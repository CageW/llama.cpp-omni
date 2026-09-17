#include "session.h"
#include "omni.h"

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// ============================================================================
// Internal helpers
// ============================================================================

static double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

static std::string make_uuid() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dist;
    uint64_t a = dist(gen);
    uint64_t b = dist(gen);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << a << std::setw(16) << b;
    return oss.str();
}

// ============================================================================
// SessionPool
// ============================================================================

void SessionPool::init_slots(std::vector<omni_context *> contexts) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!slots_.empty()) {
        throw std::logic_error("SessionPool::init_slots called twice");
    }
    slots_.reserve(contexts.size());
    for (int i = 0; i < (int)contexts.size(); ++i) {
        OmniSlot s;
        s.slot_idx  = i;
        s.state     = SlotState::IDLE;
        s.octx      = contexts[i];
        s.owns_octx = (i == 0); // root slot owns the weights
        slots_.push_back(s);
    }
}

OmniSlot * SessionPool::checkout_slot() {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [this] {
        if (stopped_) return true;
        for (auto & s : slots_) {
            if (s.state == SlotState::IDLE) return true;
        }
        return false;
    });
    if (stopped_) return nullptr;
    for (auto & s : slots_) {
        if (s.state == SlotState::IDLE) {
            s.state = SlotState::ACTIVE;
            return &s;
        }
    }
    return nullptr; // shouldn't reach
}

void SessionPool::return_slot(int idx) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (idx < 0 || idx >= (int)slots_.size()) return;
        slots_[idx].state = SlotState::IDLE;
    }
    cv_.notify_one();
}

omni_context * SessionPool::get_octx(int slot_idx) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (slot_idx < 0 || slot_idx >= (int)slots_.size()) return nullptr;
    return slots_[slot_idx].octx;
}

void SessionPool::shutdown() {
    std::vector<omni_context *> to_free;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stopped_ = true;
        for (auto & s : slots_) {
            // Only free contexts that own their weights (the root).
            // Clones share model weights; omni_free on them releases per-session
            // state (llama_context, KV caches, worker threads) without touching
            // the model, so it's safe to free them too.
            if (s.octx) {
                to_free.push_back(s.octx);
                s.octx = nullptr;
            }
        }
    }
    cv_.notify_all();
    // Free all contexts outside the lock.  omni_free joins inference threads,
    // which can take a moment, so we want to avoid holding mtx_ while it runs.
    // Free clones first (reverse order), then root last.
    for (int i = (int)to_free.size() - 1; i >= 0; --i) {
        omni_free(to_free[i]);
    }
}

// ============================================================================
// SessionManager
// ============================================================================

SessionManager::~SessionManager() {
    shutdown();
}

void SessionManager::set_pool(SessionPool * pool) {
    std::lock_guard<std::mutex> lk(mtx_);
    pool_ = pool;
}

std::string SessionManager::generate_uuid() {
    return make_uuid();
}

// ---- pool-mode methods ----

std::string SessionManager::register_slot_session(int slot_idx, omni_context * octx) {
    auto session  = std::make_unique<OmniSession>();
    session->session_id  = generate_uuid();
    session->state       = SessionState::ACTIVE;
    session->octx        = octx;
    session->owns_octx   = false; // pool owns lifetime
    session->created_at  = now_seconds();

    std::string id = session->session_id;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        pool_sessions_[id]  = std::move(session);
        session_slot_[id]   = slot_idx;
    }
    return id;
}

OmniSession * SessionManager::get(const std::string & session_id) {
    std::lock_guard<std::mutex> lk(mtx_);

    // Pool-mode lookup first.
    auto it = pool_sessions_.find(session_id);
    if (it != pool_sessions_.end()) {
        return it->second.get();
    }

    // Legacy single-session fallback.
    if (active_ && active_->session_id == session_id) {
        return active_.get();
    }
    return nullptr;
}

void SessionManager::set_close_callback(const std::string & session_id, std::function<void()> cb) {
    std::lock_guard<std::mutex> lk(mtx_);

    auto it = pool_sessions_.find(session_id);
    if (it != pool_sessions_.end()) {
        it->second->close_ws = std::move(cb);
        return;
    }
    if (active_ && active_->session_id == session_id) {
        active_->close_ws = std::move(cb);
    }
}

void SessionManager::request_transport_close(const std::string & session_id) {
    std::function<void()> close_ws;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = pool_sessions_.find(session_id);
        if (it != pool_sessions_.end()) {
            close_ws = it->second->close_ws;
        } else if (active_ && active_->session_id == session_id) {
            close_ws = active_->close_ws;
        }
    }
    if (close_ws) close_ws();
}

void SessionManager::close(const std::string & session_id) {
    int slot_to_return = -1;
    {
        std::lock_guard<std::mutex> lk(mtx_);

        // Pool-mode: unregister session and note which slot to return.
        auto it = pool_sessions_.find(session_id);
        if (it != pool_sessions_.end()) {
            auto sit = session_slot_.find(session_id);
            if (sit != session_slot_.end()) {
                slot_to_return = sit->second;
                session_slot_.erase(sit);
            }
            pool_sessions_.erase(it);
        }

        // Legacy single-session cleanup.
        if (active_ && active_->session_id == session_id) {
            // Never free the shared_octx from here; server owns its lifetime.
            active_->owns_octx = false;
            active_.reset();
        }
    }

    // Return the slot outside the lock.
    if (pool_ && slot_to_return >= 0) {
        pool_->return_slot(slot_to_return);
    }
}

int SessionManager::on_disconnect(const std::string & session_id) {
    int slot_to_return = -1;
    {
        std::lock_guard<std::mutex> lk(mtx_);

        auto it = pool_sessions_.find(session_id);
        if (it != pool_sessions_.end()) {
            auto sit = session_slot_.find(session_id);
            if (sit != session_slot_.end()) {
                slot_to_return = sit->second;
                session_slot_.erase(sit);
            }
            pool_sessions_.erase(it);
        }

        if (active_ && active_->session_id == session_id) {
            active_->owns_octx = false;
            active_.reset();
        }
    }

    if (pool_ && slot_to_return >= 0) {
        pool_->return_slot(slot_to_return);
    }
    return slot_to_return;
}

// ---- legacy single-session methods ----

std::string SessionManager::allocate() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (active_ && active_->state != SessionState::CLOSED) {
        return "";
    }
    auto session = std::make_unique<OmniSession>();
    session->session_id = generate_uuid();
    session->state      = SessionState::UNINITIALIZED;
    session->created_at = now_seconds();
    std::string id = session->session_id;
    active_ = std::move(session);
    return id;
}

bool SessionManager::activate(const std::string & session_id, omni_context * octx, bool owns_octx) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!active_ || active_->session_id != session_id) return false;
    if (active_->state != SessionState::UNINITIALIZED)  return false;
    active_->octx      = octx;
    active_->owns_octx = owns_octx;
    active_->state     = SessionState::ACTIVE;
    return true;
}

bool SessionManager::has_active() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return active_ && active_->state == SessionState::ACTIVE;
}

void SessionManager::shutdown() {
    std::unique_ptr<OmniSession> legacy;
    std::unordered_map<std::string, std::unique_ptr<OmniSession>> pool_sessions;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        legacy = std::move(active_);
        pool_sessions = std::move(pool_sessions_);
        session_slot_.clear();
    }
    // pool contexts are freed by SessionPool::shutdown(); don't double-free.
    if (legacy && legacy->owns_octx && legacy->octx) {
        omni_free(legacy->octx);
    }
}
