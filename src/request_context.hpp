#pragma once
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>
#include <memory>

// ─── Sampling parameters ────────────────────────────────────────────────────

struct SamplingParams {
    float       temperature     = 0.8f;
    float       top_p           = 0.95f;
    int         top_k           = 40;
    float       repeat_penalty  = 1.1f;
    int         max_tokens      = 512;
    std::vector<std::string> stop_sequences;

    // Mirostat v2
    bool  use_mirostat  = false;
    float mirostat_tau  = 5.0f;
    float mirostat_eta  = 0.1f;
};

// ─── Message for chat completions ───────────────────────────────────────────

struct ChatMessage {
    std::string role;     // "system" | "user" | "assistant"
    std::string content;
};

// ─── Request status ──────────────────────────────────────────────────────────

enum class RequestStatus {
    Queued,
    Prefilling,
    Decoding,
    Completed,
    Failed,
    Cancelled
};

// ─── Token channel: lock-free single-producer, single-consumer ───────────────

struct TokenChannel {
    std::queue<std::string>     tokens;
    std::mutex                  mtx;
    std::condition_variable     cv;
    bool                        done    = false;
    std::string                 error;

    void push(const std::string& tok) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            tokens.push(tok);
        }
        cv.notify_one();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            done = true;
        }
        cv.notify_all();
    }

    void fail(const std::string& msg) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            error = msg;
            done  = true;
        }
        cv.notify_all();
    }

    // Returns false when done and queue is empty
    bool pop(std::string& out, int timeout_ms = 5000) {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                    [&]{ return !tokens.empty() || done; });
        if (!tokens.empty()) {
            out = tokens.front();
            tokens.pop();
            return true;
        }
        return false;
    }
};

// ─── Inference request ───────────────────────────────────────────────────────

struct InferenceRequest {
    uint64_t                        id;
    std::string                     prompt;          // Rendered prompt string
    SamplingParams                  params;
    bool                            stream = false;

    // Internals (set by scheduler)
    std::vector<int32_t>            input_tokens;
    std::vector<int32_t>            output_tokens;
    int                             kv_slot      = -1;
    std::atomic<RequestStatus>      status       { RequestStatus::Queued };

    std::shared_ptr<TokenChannel>   channel;
    std::string                     full_text;

    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point started_at;
    std::chrono::steady_clock::time_point finished_at;

    int n_tokens_generated = 0;

    explicit InferenceRequest(uint64_t id)
        : id(id), channel(std::make_shared<TokenChannel>()),
          created_at(std::chrono::steady_clock::now()) {}
};

using RequestPtr = std::shared_ptr<InferenceRequest>;

// ─── Naive request queue ─────────────────────────────────────────────────────
// Simple vector + head index. No condition variable, no fine-grained locking.
// One mutex guards all operations. Good enough for moderate concurrency.

class RequestQueue {
public:
    void enqueue(RequestPtr req) {
        std::lock_guard<std::mutex> lk(mtx_);
        items_.push_back(std::move(req));
    }

    bool try_dequeue(RequestPtr& out) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (head_ >= items_.size()) return false;
        out = std::move(items_[head_++]);
        // Compact the vector once the dead prefix gets large
        if (head_ > 64) {
            items_.erase(items_.begin(), items_.begin() + head_);
            head_ = 0;
        }
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return items_.size() - head_;
    }

private:
    mutable std::mutex          mtx_;
    std::vector<RequestPtr>     items_;
    size_t                      head_ = 0;
};

// ─── Model/server info ───────────────────────────────────────────────────────

struct ModelInfo {
    std::string path;
    std::string name;
    int         n_ctx         = 4096;
    int         n_batch       = 512;
    int         max_batch_seq = 16;    // Max sequences in one decode step
    int         n_threads     = 4;
    int         n_gpu_layers  = 0;
    bool        use_mmap      = true;
    bool        use_mlock     = false;
    int         port          = 8080;
    std::string host          = "0.0.0.0";
    bool        verbose       = false;
};
