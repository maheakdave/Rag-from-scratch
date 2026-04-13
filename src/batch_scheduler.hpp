#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <algorithm>
// #include "llama.h"
#include "request_context.hpp"



class KvSlotManager {
public:
    explicit KvSlotManager(int n_slots) : n_slots_(n_slots), used_(n_slots, false) {}

    int allocate() {
        for (int i = 0; i < n_slots_; ++i) {
            if (!used_[i]) { used_[i] = true; return i; }
        }
        return -1; 
    }

    void release(int slot) {
        if (slot >= 0 && slot < n_slots_) used_[slot] = false;
    }

    int free_count() const {
        return (int)std::count(used_.begin(), used_.end(), false);
    }

private:
    int              n_slots_;
    std::vector<bool> used_;
};


class ContinuousBatchScheduler {
public:
    ContinuousBatchScheduler(llama_model* model, llama_context* ctx,
                             RequestQueue& queue, const ModelInfo& cfg)
        : model_(model), ctx_(ctx), queue_(queue), cfg_(cfg),
          kv_slots_(cfg.max_batch_seq)
    {
        llama_batch_init_helper();
    }

    ~ContinuousBatchScheduler() {
        stop();
        llama_batch_free(batch_);
    }

    void start() {
        running_ = true;
        worker_  = std::thread(&ContinuousBatchScheduler::run_loop, this);
    }

    void stop() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }

    struct Stats {
        uint64_t requests_completed = 0;
        uint64_t tokens_generated   = 0;
        uint64_t tokens_per_second  = 0;
        int      active_sequences   = 0;
        int      queued_requests    = 0;
    };

    Stats get_stats() const { return stats_; }

private:
    
    struct Sequence {
        RequestPtr  req;
        int         kv_slot;
        int         n_past     = 0;   // Tokens already in KV cache
        bool        prefilling = true; // First batch = prefill
        llama_sampling_context* sampling_ctx = nullptr;

        ~Sequence() {
            if (sampling_ctx) llama_sampling_free(sampling_ctx);
        }
    };



    void llama_batch_init_helper() {
        
        batch_ = llama_batch_init(cfg_.n_batch, 0, cfg_.max_batch_seq);
    }

    static void batch_add(llama_batch& batch, llama_token token,
                          int pos, std::vector<llama_seq_id> seq_ids, bool logits)
    {
        batch.token   [batch.n_tokens] = token;
        batch.pos     [batch.n_tokens] = pos;
        batch.n_seq_id[batch.n_tokens] = (int)seq_ids.size();
        for (size_t i = 0; i < seq_ids.size(); ++i)
            batch.seq_id[batch.n_tokens][i] = seq_ids[i];
        batch.logits  [batch.n_tokens] = logits;
        batch.n_tokens++;
    }



    std::vector<llama_token> tokenise(const std::string& text, bool add_bos) {
        const int n_max = text.size() + 128;
        std::vector<llama_token> tokens(n_max);
        int n = llama_tokenize(model_, text.c_str(), (int)text.size(),
                               tokens.data(), n_max, add_bos, false);
        if (n < 0) return {};
        tokens.resize(n);
        return tokens;
    }



    bool should_stop(const Sequence& seq) {
        const auto& req   = seq.req;
        const auto& stops = req->params.stop_sequences;

        // Max tokens
        if (req->n_tokens_generated >= req->params.max_tokens) return true;
        // EOS
        if (!req->output_tokens.empty() &&
            req->output_tokens.back() == llama_token_eos(model_)) return true;
        // Stop strings
        for (const auto& s : stops) {
            if (req->full_text.size() >= s.size() &&
                req->full_text.compare(req->full_text.size() - s.size(),
                                       s.size(), s) == 0) return true;
        }
        return false;
    }

    // ── Admission control: try to add new requests ────────────────────────────

    void admit_new_requests() {
        while ((int)active_.size() < cfg_.max_batch_seq) {
            int slot = kv_slots_.allocate();
            if (slot < 0) break;                    // KV cache full

            RequestPtr req;
            if (!queue_.try_dequeue(req)) {
                kv_slots_.release(slot);
                break;                              // Nothing in queue
            }
            if (req->status == RequestStatus::Cancelled) {
                kv_slots_.release(slot);
                continue;
            }

            // Tokenise
            req->input_tokens = tokenise(req->prompt, /*add_bos=*/true);
            if (req->input_tokens.empty()) {
                kv_slots_.release(slot);
                req->channel->fail("Failed to tokenise prompt");
                continue;
            }

            // Context length guard
            if ((int)req->input_tokens.size() >= cfg_.n_ctx - 4) {
                kv_slots_.release(slot);
                req->channel->fail("Prompt exceeds context window");
                continue;
            }

            req->status     = RequestStatus::Prefilling;
            req->kv_slot    = slot;
            req->started_at = std::chrono::steady_clock::now();

            llama_sampling_params sparams;
            sparams.temp             = req->params.temperature;
            sparams.top_p            = req->params.top_p;
            sparams.top_k            = req->params.top_k;
            sparams.penalty_repeat   = req->params.repeat_penalty;
            if (req->params.use_mirostat) {
                sparams.mirostat     = 2;
                sparams.mirostat_tau = req->params.mirostat_tau;
                sparams.mirostat_eta = req->params.mirostat_eta;
            }

            auto seq      = std::make_unique<Sequence>();
            seq->req      = req;
            seq->kv_slot  = slot;
            seq->sampling_ctx = llama_sampling_init(sparams);
            active_.push_back(std::move(seq));

            if (cfg_.verbose)
                std::cout << "[scheduler] admitted req " << req->id
                          << " (" << req->input_tokens.size() << " tokens)\n";
        }
    }

    // ── Build batch for one decode step ──────────────────────────────────────

    void build_batch() {
        batch_.n_tokens = 0;

        for (auto& seq : active_) {
            if (seq->prefilling) {
                // Add all prompt tokens; only last gets logits
                for (int i = 0; i < (int)seq->req->input_tokens.size(); ++i) {
                    bool want_logits = (i == (int)seq->req->input_tokens.size() - 1);
                    batch_add(batch_, seq->req->input_tokens[i],
                              seq->n_past + i,
                              { seq->kv_slot }, want_logits);
                }
            } else {
                // Add the single last generated token
                batch_add(batch_,
                          seq->req->output_tokens.empty()
                              ? seq->req->input_tokens.back()
                              : seq->req->output_tokens.back(),
                          seq->n_past,
                          { seq->kv_slot }, true);
            }
        }
    }

    // ── Sample next token for each sequence ──────────────────────────────────

    void sample_and_emit() {
        int logit_offset = 0;

        for (auto& seq : active_) {
            // Find which token index produced logits for this seq
            // (last token of its range in the batch)
            int token_idx = -1;
            for (int i = batch_.n_tokens - 1; i >= 0; --i) {
                bool belongs = false;
                for (int j = 0; j < batch_.n_seq_id[i]; ++j)
                    if (batch_.seq_id[i][j] == seq->kv_slot) { belongs = true; break; }
                if (belongs && batch_.logits[i]) { token_idx = i; break; }
            }
            if (token_idx < 0) continue;

            llama_token tok = llama_sampling_sample(seq->sampling_ctx, ctx_, nullptr, token_idx);
            llama_sampling_accept(seq->sampling_ctx, ctx_, tok, true);

            // Decode token to string
            std::vector<char> piece(32);
            int n = llama_token_to_piece(model_, tok, piece.data(), (int)piece.size(), false);
            if (n < 0) { piece.resize(-n); llama_token_to_piece(model_, tok, piece.data(), (int)piece.size(), false); n = -n; }
            std::string text(piece.data(), n);

            seq->req->output_tokens.push_back(tok);
            seq->req->full_text += text;
            seq->req->n_tokens_generated++;

            // Stream to caller
            if (seq->req->stream) seq->req->channel->push(text);

            // Update n_past
            if (seq->prefilling) {
                seq->n_past      += (int)seq->req->input_tokens.size();
                seq->prefilling   = false;
                seq->req->status  = RequestStatus::Decoding;
            } else {
                seq->n_past++;
            }

            stats_.tokens_generated++;
            (void)logit_offset;
        }
    }

    // ── Remove finished sequences ─────────────────────────────────────────────

    void retire_finished() {
        auto it = active_.begin();
        while (it != active_.end()) {
            auto& seq = *it;
            if (should_stop(*seq) ||
                seq->req->status == RequestStatus::Cancelled)
            {
                // KV cache cleanup: remove this sequence's entries
                llama_kv_cache_seq_rm(ctx_, seq->kv_slot, -1, -1);
                kv_slots_.release(seq->kv_slot);

                seq->req->finished_at = std::chrono::steady_clock::now();
                seq->req->status      = RequestStatus::Completed;

                // Send accumulated text for non-streaming callers
                if (!seq->req->stream)
                    seq->req->channel->push(seq->req->full_text);
                seq->req->channel->finish();

                stats_.requests_completed++;
                it = active_.erase(it);

                if (cfg_.verbose)
                    std::cout << "[scheduler] finished req " << seq->req->id
                              << " (" << seq->req->n_tokens_generated << " tok)\n";
            } else {
                ++it;
            }
        }
    }

    // ── Main loop ─────────────────────────────────────────────────────────────

    void run_loop() {
        auto last_stats = std::chrono::steady_clock::now();
        uint64_t last_tokens = 0;

        while (running_) {
            // 1. Try to admit waiting requests
            admit_new_requests();

            if (active_.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // 2. Build the mixed prefill+decode batch
            build_batch();

            if (batch_.n_tokens == 0) continue;

            // 3. Single forward pass for the whole batch
            if (llama_decode(ctx_, batch_) != 0) {
                std::cerr << "[scheduler] llama_decode failed\n";
                // Fail all active sequences
                for (auto& s : active_) s->req->channel->fail("Decode error");
                active_.clear();
                continue;
            }

            // 4. Sample + emit tokens
            sample_and_emit();

            // 5. Remove done sequences
            retire_finished();

            // 6. Update stats
            stats_.active_sequences = (int)active_.size();
            stats_.queued_requests  = (int)queue_.size();
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<double>(now - last_stats).count();
            if (elapsed > 1.0) {
                stats_.tokens_per_second =
                    (uint64_t)((stats_.tokens_generated - last_tokens) / elapsed);
                last_tokens = stats_.tokens_generated;
                last_stats  = now;
            }
        }

        // Drain active sequences on shutdown
        for (auto& s : active_) {
            if (!s->req->stream) s->req->channel->push(s->req->full_text);
            s->req->channel->fail("Server shutting down");
        }
    }

    // ── Members ───────────────────────────────────────────────────────────────

    llama_model*    model_;
    llama_context*  ctx_;
    RequestQueue&   queue_;
    const ModelInfo cfg_;

    KvSlotManager   kv_slots_;
    llama_batch     batch_;

    std::vector<std::unique_ptr<Sequence>> active_;
    std::thread     worker_;
    std::atomic<bool> running_ { false };

    mutable Stats   stats_;
};
