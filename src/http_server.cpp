

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <atomic>
#include <signal.h>


#include "libs/httplib.h"        
#include "libs/nlohmann/json.hpp"
#include "libs/llama.h"
#include "libs/llama-sampling.h" 

#include "request_context.hpp"
#include "batch_scheduler.hpp"

using json = nlohmann::json;



static std::atomic<bool> g_shutdown { false };
static std::atomic<uint64_t> g_req_id { 0 };


static uint64_t next_id() { return ++g_req_id; }

static void set_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

static void json_error(httplib::Response& res, int code, const std::string& msg) {
    set_cors(res);
    res.status = code;
    res.set_content(json{{"error", {{"message", msg}, {"code", code}}}}.dump(), "application/json");
}

// Parse SamplingParams from request JSON
static SamplingParams parse_sampling(const json& j) {
    SamplingParams p;
    if (j.contains("temperature"))     p.temperature    = j["temperature"].get<float>();
    if (j.contains("top_p"))           p.top_p          = j["top_p"].get<float>();
    if (j.contains("top_k"))           p.top_k          = j["top_k"].get<int>();
    if (j.contains("repetition_penalty")) p.repeat_penalty = j["repetition_penalty"].get<float>();
    if (j.contains("max_tokens"))      p.max_tokens     = j["max_tokens"].get<int>();
    if (j.contains("stop")) {
        const auto& s = j["stop"];
        if (s.is_string())        p.stop_sequences = { s.get<std::string>() };
        else if (s.is_array())    p.stop_sequences = s.get<std::vector<std::string>>();
    }
    return p;
}

// Apply a simple chat template (ChatML format)
// llama.cpp's llama_apply_chat_template() can be used instead if the model has one
static std::string apply_chat_template(const std::vector<ChatMessage>& messages) {
    std::ostringstream ss;
    for (const auto& m : messages) {
        ss << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
    }
    ss << "<|im_start|>assistant\n";
    return ss.str();
}

// Try to use the model's built-in chat template, fall back to ChatML
static std::string apply_chat_template_model(llama_model* model,
                                              const std::vector<ChatMessage>& msgs) {
    // Convert to llama_chat_message array
    std::vector<llama_chat_message> chat;
    std::vector<std::string> roles_storage, content_storage;
    for (const auto& m : msgs) {
        roles_storage.push_back(m.role);
        content_storage.push_back(m.content);
    }
    for (size_t i = 0; i < msgs.size(); ++i) {
        chat.push_back({ roles_storage[i].c_str(), content_storage[i].c_str() });
    }

    std::vector<char> buf(4096);
    int n = llama_apply_chat_template(model, nullptr,
                                      chat.data(), chat.size(),
                                      /*add_ass=*/true,
                                      buf.data(), (int)buf.size());
    if (n > 0) {
        if (n > (int)buf.size()) { buf.resize(n); llama_apply_chat_template(model, nullptr, chat.data(), chat.size(), true, buf.data(), n); }
        return std::string(buf.data(), n);
    }
    // Fallback
    return apply_chat_template(msgs);
}

// ─── SSE Streaming helper ─────────────────────────────────────────────────────

// Streams tokens to the client as Server-Sent Events in OpenAI delta format
static void stream_completion(httplib::Response& res,
                               RequestPtr req,
                               const std::string& model_name)
{
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection",    "keep-alive");
    set_cors(res);

    const std::string req_id = "chatcmpl-" + std::to_string(req->id);

    res.set_chunked_content_provider("text/event-stream",
        [req, req_id, model_name](size_t /*offset*/, httplib::DataSink& sink) -> bool
    {
        std::string tok;
        while (req->channel->pop(tok, 5000)) {
            // Build delta event
            json delta = {
                {"id",      req_id},
                {"object",  "chat.completion.chunk"},
                {"model",   model_name},
                {"choices", json::array({
                    {{"index", 0},
                     {"delta", {{"content", tok}}},
                     {"finish_reason", nullptr}}
                })}
            };
            std::string data = "data: " + delta.dump() + "\n\n";
            if (!sink.write(data.data(), data.size())) return false;
        }

        // Final [DONE] event
        std::string done_event = "data: [DONE]\n\n";
        sink.write(done_event.data(), done_event.size());
        return true;
    });
}

// ─── Route: POST /v1/chat/completions ────────────────────────────────────────

static void route_chat_completions(const httplib::Request& req, httplib::Response& res,
                                   RequestQueue& queue, llama_model* model) {
    json body;
    try { body = json::parse(req.body); }
    catch (...) { json_error(res, 400, "Invalid JSON"); return; }

    if (!body.contains("messages") || !body["messages"].is_array()) {
        json_error(res, 400, "'messages' array required"); return;
    }

    // Parse messages
    std::vector<ChatMessage> messages;
    for (const auto& m : body["messages"]) {
        if (!m.contains("role") || !m.contains("content")) continue;
        messages.push_back({ m["role"].get<std::string>(), m["content"].get<std::string>() });
    }

    auto request       = std::make_shared<InferenceRequest>(next_id());
    request->params    = parse_sampling(body);
    request->stream    = body.value("stream", false);
    request->prompt    = apply_chat_template_model(model, messages);

    queue.enqueue(request);

    std::string model_name = body.value("model", "llama");

    if (request->stream) {
        stream_completion(res, request, model_name);
        return;
    }

    // Non-streaming: wait for the full response
    std::string output;
    request->channel->pop(output, 60000);

    if (!request->channel->error.empty()) {
        json_error(res, 500, request->channel->error); return;
    }

    // Strip stop sequences from output
    for (const auto& stop : request->params.stop_sequences) {
        auto pos = output.find(stop);
        if (pos != std::string::npos) output = output.substr(0, pos);
    }

    auto elapsed = std::chrono::duration<double>(
        request->finished_at - request->started_at).count();

    json resp = {
        {"id",      "chatcmpl-" + std::to_string(request->id)},
        {"object",  "chat.completion"},
        {"model",   model_name},
        {"choices", json::array({
            {{"index", 0},
             {"message", {{"role", "assistant"}, {"content", output}}},
             {"finish_reason", "stop"}}
        })},
        {"usage", {
            {"prompt_tokens",     (int)request->input_tokens.size()},
            {"completion_tokens", request->n_tokens_generated},
            {"total_tokens",      (int)request->input_tokens.size() + request->n_tokens_generated}
        }}
    };

    set_cors(res);
    res.set_content(resp.dump(), "application/json");
}

// ─── Route: POST /v1/completions ──────────────────────────────────────────────

static void route_completions(const httplib::Request& req, httplib::Response& res,
                              RequestQueue& queue, llama_model* /*model*/) {
    json body;
    try { body = json::parse(req.body); }
    catch (...) { json_error(res, 400, "Invalid JSON"); return; }

    std::string prompt;
    if (body.contains("prompt")) {
        if (body["prompt"].is_string()) prompt = body["prompt"].get<std::string>();
        else { json_error(res, 400, "'prompt' must be a string"); return; }
    } else {
        json_error(res, 400, "'prompt' required"); return;
    }

    auto request       = std::make_shared<InferenceRequest>(next_id());
    request->params    = parse_sampling(body);
    request->stream    = body.value("stream", false);
    request->prompt    = prompt;

    queue.enqueue(request);

    std::string model_name = body.value("model", "llama");

    if (request->stream) {
        // Stream plain completions
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        set_cors(res);
        const std::string req_id = "cmpl-" + std::to_string(request->id);

        res.set_chunked_content_provider("text/event-stream",
            [request, req_id, model_name](size_t, httplib::DataSink& sink) -> bool
        {
            std::string tok;
            while (request->channel->pop(tok, 5000)) {
                json ev = {
                    {"id", req_id}, {"object", "text_completion.chunk"},
                    {"model", model_name},
                    {"choices", json::array({
                        {{"index", 0}, {"text", tok}, {"finish_reason", nullptr}}
                    })}
                };
                std::string data = "data: " + ev.dump() + "\n\n";
                if (!sink.write(data.data(), data.size())) return false;
            }
            sink.write("data: [DONE]\n\n", 14);
            return true;
        });
        return;
    }

    std::string output;
    request->channel->pop(output, 60000);

    if (!request->channel->error.empty()) {
        json_error(res, 500, request->channel->error); return;
    }

    json resp = {
        {"id",      "cmpl-" + std::to_string(request->id)},
        {"object",  "text_completion"},
        {"model",   model_name},
        {"choices", json::array({
            {{"index", 0}, {"text", output}, {"finish_reason", "stop"}}
        })},
        {"usage", {
            {"prompt_tokens",     (int)request->input_tokens.size()},
            {"completion_tokens", request->n_tokens_generated},
            {"total_tokens",      (int)request->input_tokens.size() + request->n_tokens_generated}
        }}
    };

    set_cors(res);
    res.set_content(resp.dump(), "application/json");
}

// ─── Route: GET /v1/models ───────────────────────────────────────────────────

static void route_models(const httplib::Request& /*req*/, httplib::Response& res,
                         const ModelInfo& info) {
    set_cors(res);
    json resp = {
        {"object", "list"},
        {"data", json::array({
            {{"id", info.name}, {"object", "model"}, {"owned_by", "local"}}
        })}
    };
    res.set_content(resp.dump(), "application/json");
}

// ─── Route: GET /metrics ─────────────────────────────────────────────────────

static void route_metrics(const httplib::Request& /*req*/, httplib::Response& res,
                          ContinuousBatchScheduler& sched) {
    set_cors(res);
    auto s = sched.get_stats();
    json resp = {
        {"requests_completed", s.requests_completed},
        {"tokens_generated",   s.tokens_generated},
        {"tokens_per_second",  s.tokens_per_second},
        {"active_sequences",   s.active_sequences},
        {"queued_requests",    s.queued_requests}
    };
    res.set_content(resp.dump(), "application/json");
}

// ─── CLI argument parsing ─────────────────────────────────────────────────────

static void print_usage(const char* argv0) {
    std::cout <<
        "Usage: " << argv0 << " [options]\n"
        "\nOptions:\n"
        "  -m, --model    PATH    Path to GGUF model file (required)\n"
        "  --host         ADDR    Bind host (default: 0.0.0.0)\n"
        "  --port         PORT    Listen port (default: 8080)\n"
        "  -c, --ctx-size N       Context window size (default: 4096)\n"
        "  --batch-size   N       Prompt processing batch size (default: 512)\n"
        "  --max-seqs     N       Max concurrent sequences (default: 16)\n"
        "  -t, --threads  N       CPU threads (default: 4)\n"
        "  -ngl           N       GPU layers to offload (default: 0)\n"
        "  --no-mmap             Disable memory-mapped I/O\n"
        "  --mlock               Lock model in RAM\n"
        "  -v, --verbose         Verbose output\n"
        "  -h, --help            Show this help\n";
}

static ModelInfo parse_args(int argc, char** argv) {
    ModelInfo cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing argument for " << a << "\n"; exit(1); }
            return argv[++i];
        };
        if      (a == "-m"  || a == "--model")      cfg.path        = next();
        else if (a == "--host")                      cfg.host        = next();
        else if (a == "--port")                      cfg.port        = std::stoi(next());
        else if (a == "-c"  || a == "--ctx-size")    cfg.n_ctx       = std::stoi(next());
        else if (a == "--batch-size")                cfg.n_batch     = std::stoi(next());
        else if (a == "--max-seqs")                  cfg.max_batch_seq = std::stoi(next());
        else if (a == "-t"  || a == "--threads")     cfg.n_threads   = std::stoi(next());
        else if (a == "-ngl")                        cfg.n_gpu_layers = std::stoi(next());
        else if (a == "--no-mmap")                   cfg.use_mmap    = false;
        else if (a == "--mlock")                     cfg.use_mlock   = true;
        else if (a == "-v"  || a == "--verbose")     cfg.verbose     = true;
        else if (a == "-h"  || a == "--help")        { print_usage(argv[0]); exit(0); }
        else { std::cerr << "Unknown option: " << a << "\n"; exit(1); }
    }
    if (cfg.path.empty()) {
        std::cerr << "Error: --model is required\n\n";
        print_usage(argv[0]);
        exit(1);
    }
    // Derive a friendly model name from the filename
    size_t sep = cfg.path.find_last_of("/\\");
    cfg.name = (sep != std::string::npos) ? cfg.path.substr(sep + 1) : cfg.path;
    auto dot = cfg.name.rfind('.');
    if (dot != std::string::npos) cfg.name = cfg.name.substr(0, dot);
    return cfg;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    ModelInfo cfg = parse_args(argc, argv);

    // ── Load model ──────────────────────────────────────────────────────────

    std::cout << "[server] Loading model: " << cfg.path << "\n";
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;
    model_params.use_mmap     = cfg.use_mmap;
    model_params.use_mlock    = cfg.use_mlock;

    llama_model* model = llama_load_model_from_file(cfg.path.c_str(), model_params);
    if (!model) {
        std::cerr << "[server] Failed to load model from " << cfg.path << "\n";
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx      = cfg.n_ctx;
    ctx_params.n_batch    = cfg.n_batch;
    ctx_params.n_threads  = cfg.n_threads;
    // Allocate KV cache for max_batch_seq parallel sequences
    ctx_params.n_seq_max  = cfg.max_batch_seq;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "[server] Failed to create llama context\n";
        llama_free_model(model);
        return 1;
    }

    std::cout << "[server] Model loaded. Context: " << cfg.n_ctx
              << " tokens, max sequences: " << cfg.max_batch_seq << "\n";

    // ── Start scheduler ──────────────────────────────────────────────────────

    RequestQueue queue;
    ContinuousBatchScheduler scheduler(model, ctx, queue, cfg);
    scheduler.start();

    // ── Configure HTTP server ────────────────────────────────────────────────

    httplib::Server svr;
    svr.set_read_timeout(120);
    svr.set_write_timeout(120);
    svr.set_idle_interval(0, 100000); // 100ms

    // CORS preflight
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        set_cors(res); res.status = 204;
    });

    // ── Endpoints ────────────────────────────────────────────────────────────

    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        route_chat_completions(req, res, queue, model);
    });

    svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        route_completions(req, res, queue, model);
    });

    svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        route_models(req, res, cfg);
    });

    svr.Get("/metrics", [&](const httplib::Request& req, httplib::Response& res) {
        route_metrics(req, res, scheduler);
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // ── Signal handler ───────────────────────────────────────────────────────

    signal(SIGINT,  [](int) { g_shutdown = true; });
    signal(SIGTERM, [](int) { g_shutdown = true; });

    std::thread shutdown_watcher([&]() {
        while (!g_shutdown) std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "\n[server] Shutting down...\n";
        svr.stop();
    });

    std::cout << "[server] Listening on http://" << cfg.host << ":" << cfg.port << "\n"
              << "[server] Endpoints:\n"
              << "  POST /v1/chat/completions\n"
              << "  POST /v1/completions\n"
              << "  GET  /v1/models\n"
              << "  GET  /metrics\n"
              << "  GET  /health\n";

    svr.listen(cfg.host, cfg.port);

    shutdown_watcher.join();
    scheduler.stop();
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    std::cout << "[server] Clean exit.\n";
    return 0;
}
