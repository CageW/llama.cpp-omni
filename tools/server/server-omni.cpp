// Omni streaming HTTP server — standalone omni API endpoints
// Based on the old server.cpp omni handlers, adapted for the new llama.cpp APIs

#include "omni.h"
#include "runtime-autotune.h"
#include "runtime-autotune-runner.h"
#include "runtime-profile.h"
#include "llama.h"
#include "build-info.h"
#include "common.h"
#include "log.h"
#include "arg.h"
#include "sampling.h"
#include "session.h"
#include "ws_handler.h"

#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <iomanip>
#include <map>
#include <optional>
#include <atomic>
#include <sstream>
#include <string>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static json format_error_response(const std::string & message, const std::string & type = "invalid_request_error") {
    return json{{"error", {{"message", message}, {"type", type}}}};
}

template<typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    if (body.contains(key)) {
        try {
            return body.at(key).get<T>();
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

static void res_ok(httplib::Response & res, const json & data) {
    res.set_content(data.dump(), "application/json");
}

static void res_error(httplib::Response & res, const json & err) {
    res.status = json_value(err, "code", 500);
    res.set_content(err.dump(), "application/json");
}

static bool server_sent_event(httplib::DataSink & sink, const json & ev) {
    std::string str = "data: " + ev.dump() + "\n\n";
    return sink.write(str.data(), str.size());
}

static std::string parent_dir(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static bool ensure_omni_model_paths_from_llm(common_params & params) {
    if (params.model.path.empty()) {
        return false;
    }
    const std::string root = parent_dir(params.model.path);
    if (root.empty()) {
        return false;
    }
    if (params.vpm_model.empty()) {
        params.vpm_model = root + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    }
    if (params.apm_model.empty()) {
        params.apm_model = root + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    }
    if (params.tts_model.empty()) {
        params.tts_model = root + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    }
    if (params.tts_bin_dir.empty()) {
        params.tts_bin_dir = root + "/tts";
    }
    return true;
}

static std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

static std::string filename_timestamp() {
    static std::atomic<uint64_t> sequence{0};
    std::string output = utc_timestamp();
    output.erase(std::remove_if(output.begin(), output.end(), [](char character) {
        return character == '-' || character == ':';
    }), output.end());
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    output += "-" + std::to_string(tick) + "-" + std::to_string(sequence.fetch_add(1));
    return output;
}

static omni::autotune_fixture default_autotune_fixture() {
    std::filesystem::path asset_root;
    if (const char * override_path = std::getenv("OMNI_AUTOTUNE_FIXTURE_DIR")) {
        asset_root = override_path;
    } else {
        asset_root = OMNI_ASSET_DIR;
    }

    omni::autotune_fixture fixture;
    fixture.frame_prefix = (asset_root / "test_case/duplex_omni_test_case/duplex_omni_test_case_").string();
    fixture.reference_audio = (asset_root / "default_ref_audio/default_ref_audio.wav").string();
    return fixture;
}

static std::string indexed_fixture_path(const std::string & prefix, int32_t index, const char * extension) {
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "%04d%s", index, extension);
    return prefix + suffix;
}

static std::vector<std::string> autotune_input_files(const omni::effective_runtime_config & config,
                                                     const omni::model_inventory &          inventory,
                                                     const omni::autotune_fixture &         fixture) {
    std::vector<std::string> output = {
        config.llm_model,
        inventory.llm_f16,
        inventory.llm_q8_0,
        inventory.llm_q4_k_m,
        config.vision_model,
        config.audio_model,
        config.tts_model,
        config.projector_model,
        inventory.token2wav_encoder,
        inventory.token2wav_flow_matching,
        inventory.token2wav_flow_extra,
        inventory.token2wav_hifigan,
        inventory.token2wav_prompt_cache,
        fixture.reference_audio,
    };
    for (int32_t index = 0; index < fixture.frame_count; ++index) {
        output.push_back(indexed_fixture_path(fixture.frame_prefix, index, ".wav"));
        output.push_back(indexed_fixture_path(fixture.frame_prefix, index, ".jpg"));
    }
    output.erase(std::remove(output.begin(), output.end(), std::string()), output.end());
    std::sort(output.begin(), output.end());
    output.erase(std::unique(output.begin(), output.end()), output.end());
    return output;
}

static std::string autotune_cpu_mask(const common_cpu_params & params) {
    if (!params.mask_valid) {
        return "any";
    }
    std::string output;
    output.reserve(GGML_MAX_N_THREADS);
    for (const bool enabled : params.cpumask) {
        output.push_back(enabled ? '1' : '0');
    }
    return output;
}

static std::string autotune_tensor_overrides(const common_params & params) {
    std::ostringstream output;
    for (const auto & override : params.tensor_buft_overrides) {
        if (override.pattern == nullptr) {
            break;
        }
        output << override.pattern << '='
               << (override.buft == nullptr ? "null" : ggml_backend_buft_name(override.buft)) << ';';
    }
    return output.str();
}

static std::map<std::string, std::string> autotune_runtime_inputs(const common_params & params) {
    auto boolean = [](bool value) { return value ? "true" : "false"; };
    auto environment = [](const char * name) {
        const char * value = std::getenv(name);
        return value == nullptr ? std::string() : std::string(value);
    };

    auto sampling = params.sampling;
    sampling.seed = 42;
    return {
        { "n_batch", std::to_string(params.n_batch) },
        { "n_ubatch", std::to_string(params.n_ubatch) },
        { "threads", std::to_string(params.cpuparams.n_threads) },
        { "threads_batch", std::to_string(params.cpuparams_batch.n_threads) },
        { "cpu_mask", autotune_cpu_mask(params.cpuparams) },
        { "cpu_mask_batch", autotune_cpu_mask(params.cpuparams_batch) },
        { "cpu_strict", boolean(params.cpuparams.strict_cpu) },
        { "cpu_strict_batch", boolean(params.cpuparams_batch.strict_cpu) },
        { "cpu_priority", std::to_string(static_cast<int>(params.cpuparams.priority)) },
        { "cpu_priority_batch", std::to_string(static_cast<int>(params.cpuparams_batch.priority)) },
        { "cpu_poll", std::to_string(params.cpuparams.poll) },
        { "cpu_poll_batch", std::to_string(params.cpuparams_batch.poll) },
        { "flash_attn", std::to_string(static_cast<int>(params.flash_attn_type)) },
        { "cache_type_k", std::to_string(static_cast<int>(params.cache_type_k)) },
        { "cache_type_v", std::to_string(static_cast<int>(params.cache_type_v)) },
        { "use_mmap", boolean(params.use_mmap) },
        { "use_direct_io", boolean(params.use_direct_io) },
        { "use_mlock", boolean(params.use_mlock) },
        { "no_kv_offload", boolean(params.no_kv_offload) },
        { "no_op_offload", boolean(params.no_op_offload) },
        { "no_extra_bufts", boolean(params.no_extra_bufts) },
        { "no_host", boolean(params.no_host) },
        { "tensor_overrides", autotune_tensor_overrides(params) },
        { "vision_coreml_model", params.vision_coreml_model_path },
        { "token2wav_coreml_model", params.token2wav_coreml_model_path },
        { "sampling", sampling.print() },
        { "CUDA_VISIBLE_DEVICES", environment("CUDA_VISIBLE_DEVICES") },
        { "OMNI_VOC_DEVICE", environment("OMNI_VOC_DEVICE") },
        { "GGML_CUDA_ENABLE_UNIFIED_MEMORY", environment("GGML_CUDA_ENABLE_UNIFIED_MEMORY") },
    };
}

static std::string autotune_cache_status_name(omni::autotune_cache_status status) {
    switch (status) {
        case omni::autotune_cache_status::applied: return "applied";
        case omni::autotune_cache_status::missing: return "missing";
        case omni::autotune_cache_status::stale:   return "stale";
        case omni::autotune_cache_status::invalid: return "invalid";
    }
    return "invalid";
}

static std::string autotune_cpu_thread_probes() {
    const uint32_t available = std::max(1U, std::thread::hardware_concurrency());
    std::vector<int32_t> probes;
    for (const int32_t requested : { 8, 16, 32, 64 }) {
        const int32_t value = std::min<int32_t>(requested, static_cast<int32_t>(available));
        if (std::find(probes.begin(), probes.end(), value) == probes.end()) {
            probes.push_back(value);
        }
    }
    std::ostringstream output;
    for (size_t index = 0; index < probes.size(); ++index) {
        if (index > 0) {
            output << ',';
        }
        output << probes[index];
    }
    return output.str();
}

static omni::autotune_fingerprint build_autotune_fingerprint(
    const common_params &                    params,
    const omni::hardware_snapshot &          hardware,
    const omni::effective_runtime_config &   config,
    const omni::model_inventory &            inventory,
    const omni::runtime_profile_overrides & overrides,
    const omni::autotune_fixture &           fixture,
    std::string &                            error) {
    const omni::autotune_build_identity build = {
        llama_build_number(), llama_commit(), llama_build_target(),
    };
    auto runtime_inputs = autotune_runtime_inputs(params);
    runtime_inputs["autotune_rounds"] = std::to_string(params.omni_runtime_profile.autotune_rounds);
    runtime_inputs["autotune_warmup_rounds"] =
        std::to_string(params.omni_runtime_profile.autotune_warmup_rounds);
    runtime_inputs["autotune_allow_quantized"] =
        params.omni_runtime_profile.autotune_allow_quantized ? "true" : "false";
    runtime_inputs["token2wav_threads"] = std::to_string(params.omni_runtime_profile.token2wav_threads);
    runtime_inputs["cpu_thread_probes"] = autotune_cpu_thread_probes();
    runtime_inputs["fixture_frame_count"] = std::to_string(fixture.frame_count);
    runtime_inputs["fixture_stream_interval_ms"] = std::to_string(fixture.stream_interval_ms);
    runtime_inputs["fixture_frame_timeout_ms"] = std::to_string(fixture.frame_timeout_ms);
    runtime_inputs["fixture_audio_timeout_ms"] = std::to_string(fixture.audio_timeout_ms);
    runtime_inputs["output_hash_policy"] = "sha256-response-normalized-tts-i32le-v1";
    const auto runtime_identity = omni::detect_autotune_runtime_identity(error);
    if (!error.empty()) {
        return {};
    }
    return omni::create_autotune_fingerprint(
        build, hardware, autotune_input_files(config, inventory, fixture), config.requested_profile, overrides, error,
        runtime_inputs, runtime_identity);
}

static void apply_cached_autotune_config(omni::effective_runtime_config &         config,
                                         const omni::autotune_cache_read_result & cache,
                                         const omni::hardware_snapshot &          hardware) {
    config.autotune_cache_status = autotune_cache_status_name(cache.status);
    config.autotune_cache_reason = cache.reason;
    if (cache.status != omni::autotune_cache_status::applied) {
        return;
    }
    const auto selected = std::find_if(cache.record.candidates.begin(), cache.record.candidates.end(),
                                       [&](const auto & candidate) {
                                           return candidate.id == cache.record.selected_candidate && candidate.valid;
                                       });
    if (selected == cache.record.candidates.end()) {
        config.autotune_cache_status = "invalid";
        config.autotune_cache_reason = "matching cache did not contain its selected candidate";
        return;
    }
    config = selected->config;
    for (const auto & device : hardware.devices) {
        if (device.name != config.primary_device_name) {
            continue;
        }
        config.primary_device_description  = device.description;
        config.primary_device_free_memory  = device.free_memory;
        config.primary_device_total_memory = device.total_memory;
        break;
    }
    config.reason = "matching runtime autotune cache selected measured candidate " + selected->id;
    config.autotune_cache_status        = "applied";
    config.autotune_cache_reason        = cache.reason;
    config.autotune_selected_candidate  = selected->id;
    config.output_consistent_with_baseline = selected->output_matches_baseline;
    config.audio_semantics_verified     = cache.record.audio_semantics_verified;
}

static int run_autotune_refresh(const common_params &                    params,
                                const omni::effective_runtime_config &   base_config,
                                const omni::hardware_snapshot &          hardware,
                                const omni::model_inventory &            inventory,
                                const omni::runtime_profile_overrides & overrides,
                                const omni::autotune_fixture &           fixture,
                                const std::string &                      cache_path) {
    std::string error;
    const auto fingerprint = build_autotune_fingerprint(
        params, hardware, base_config, inventory, overrides, fixture, error);
    if (!error.empty()) {
        LOG_ERR("autotune calibration failed: %s\n", error.c_str());
        return 1;
    }

    omni::autotune_plan_options plan_options;
    plan_options.inventory            = inventory;
    plan_options.model_explicit       = params.omni_runtime_profile.model_explicit;
    plan_options.allow_quantized      = params.omni_runtime_profile.autotune_allow_quantized;
    plan_options.hardware_concurrency = std::thread::hardware_concurrency();
    const auto previous_cache = omni::read_autotune_cache(cache_path, fingerprint);
    if (previous_cache.status == omni::autotune_cache_status::applied) {
        auto validation_options = plan_options;
        validation_options.previous_measured_module_bytes = previous_cache.record.memory_estimate_inputs;
        const auto previous_plan = omni::create_autotune_candidate_plan(
            base_config, hardware, overrides, validation_options);
        std::string validation_error;
        if (omni::validate_autotune_cache_against_plan(
                previous_cache.record, previous_plan, validation_error)) {
            plan_options.previous_measured_module_bytes =
                omni::autotune_measured_module_bytes(previous_cache.record.candidates);
            LOG_INF("Autotune: reusing measured module memory from the matching previous cache\n");
        } else {
            LOG_WRN("Autotune: previous cache matched the fingerprint but failed validation; "
                    "measured memory will not be reused: %s\n", validation_error.c_str());
        }
    }
    const auto plan = omni::create_autotune_candidate_plan(base_config, hardware, overrides, plan_options);
    if (plan.candidates.empty()) {
        const std::string reason = plan.skipped_reasons.empty()
                                       ? "no eligible placement candidate"
                                       : plan.skipped_reasons.front();
        LOG_ERR("autotune calibration failed: %s\n", reason.c_str());
        return 1;
    }

    omni::autotune_cache_record record;
    record.fingerprint              = fingerprint;
    record.skipped_reasons          = plan.skipped_reasons;
    record.memory_estimate_inputs   = plan_options.previous_measured_module_bytes;
    record.audio_semantics_verified = false;
    record.created_at               = utc_timestamp();

    std::filesystem::path cache_file(cache_path);
    const auto run_root = cache_file.parent_path() / "autotune-runs" / filename_timestamp();
    std::map<std::string, std::vector<std::filesystem::path>> run_directories;
    std::map<std::string, std::filesystem::path> first_failed_directory;
    auto schedule = omni::execute_autotune_schedule(
        plan, params.omni_runtime_profile.autotune_rounds,
        params.omni_runtime_profile.autotune_warmup_rounds,
        [&](const omni::autotune_candidate_record & candidate,
            const omni::autotune_schedule_request & request,
            std::string & candidate_error) {
            const auto run_directory = request.screening
                                           ? run_root / candidate.id / "screening"
                                           : run_root / candidate.id /
                                                 ("round-" + std::to_string(request.round_index));
            run_directories[candidate.id].push_back(run_directory);
            if (request.screening) {
                LOG_INF("Autotune: screening CPU candidate %s in %s\n",
                        candidate.id.c_str(), run_directory.string().c_str());
            } else {
                LOG_INF("Autotune: measuring candidate %s, round %d/%d in %s\n",
                        candidate.id.c_str(), request.round_index + 1,
                        params.omni_runtime_profile.autotune_rounds, run_directory.string().c_str());
            }
            return omni::run_autotune_candidate(
                params, candidate, fixture, run_directory.string(), candidate_error,
                request.warmup_rounds);
        });
    record.candidates = std::move(schedule.candidates);
    for (const auto & failure : schedule.first_failed_round) {
        first_failed_directory[failure.first] = failure.second < 0
                                                    ? run_root / failure.first / "screening"
                                                    : run_root / failure.first /
                                                          ("round-" + std::to_string(failure.second));
    }
    for (const auto & candidate : record.candidates) {
        if (!candidate.valid) {
            LOG_WRN("Autotune: rejected candidate %s: %s\n",
                    candidate.id.c_str(), candidate.rejection_reason.c_str());
        }
    }
    if (!schedule.output_consistency_reference_available) {
        LOG_WRN("Autotune: Phase 2 baseline did not provide valid paired output evidence\n");
    }

    const auto selection = omni::select_autotune_candidate(record.candidates, plan.baseline_candidate);
    record.selection_reason = selection.reason;
    if (!selection.ok) {
        const std::string diagnostic_path = cache_path + ".failed-" + filename_timestamp() + ".json";
        std::string diagnostic_error;
        if (!omni::write_autotune_cache(diagnostic_path, record, diagnostic_error)) {
            LOG_WRN("Autotune: could not write diagnostic report: %s\n", diagnostic_error.c_str());
        }
        LOG_ERR("autotune calibration failed: %s; failed run directories retained under %s\n",
                selection.reason.c_str(), run_root.string().c_str());
        return 1;
    }

    record.selected_candidate = selection.selected_candidate;
    if (!omni::validate_autotune_cache_against_plan(record, plan, error)) {
        LOG_ERR("autotune calibration failed self-validation: %s; run directories retained under %s\n",
                error.c_str(), run_root.string().c_str());
        return 1;
    }
    if (!omni::write_autotune_cache(cache_path, record, error)) {
        LOG_ERR("autotune calibration failed: could not replace cache: %s\n", error.c_str());
        return 1;
    }

    for (const auto & candidate : record.candidates) {
        for (const auto & run_directory : run_directories[candidate.id]) {
            const auto failed = first_failed_directory.find(candidate.id);
            if (!candidate.valid && failed != first_failed_directory.end() && failed->second == run_directory) {
                continue;
            }
            std::error_code remove_error;
            std::filesystem::remove_all(run_directory, remove_error);
            if (remove_error) {
                LOG_WRN("Autotune: could not remove completed run directory %s: %s\n",
                        run_directory.string().c_str(), remove_error.message().c_str());
            }
        }
    }

    LOG_INF("Autotune: selected %s; cache=%s; audio_semantics_verified=false\n",
            record.selected_candidate.c_str(), cache_path.c_str());
    return 0;
}

struct omni_server_state {
    omni_context * octx = nullptr;    // WS backend uses this as shared_octx
    std::mutex octx_mutex;            // protects omni_context lifecycle + prefill/decode entry
    SessionManager session_mgr;       // WS backend session management
};

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_OMNI_SERVER)) {
        return 1;
    }

    // omni HTTP server is single-session (1:1 duplex), so 1 sequence is enough.
    // common_params defaults n_parallel to -1 ("auto"); each example resolves it
    // itself (see tools/server/server.cpp). Without this, n_seq_max overflows
    // uint32 and trips LLAMA_MAX_SEQ(256) inside llama_context.
    if (params.n_parallel < 0) {
        params.n_parallel = 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    std::optional<omni::effective_runtime_config> runtime_config;
    if (!params.omni_runtime_profile.autotune.empty() && params.omni_runtime_profile.profile.empty()) {
        LOG_ERR("--autotune refresh requires --profile auto\n");
        llama_backend_free();
        return 1;
    }
    if (!params.omni_runtime_profile.profile.empty()) {
        std::string model_root = params.omni_runtime_profile.model_dir;
        if (model_root.empty() && !params.model.path.empty()) {
            model_root = parent_dir(params.model.path);
        }
        if (model_root.empty()) {
            LOG_ERR("--profile requires --model-dir or --model to locate the MiniCPM-o 4.5 model tree\n");
            llama_backend_free();
            return 1;
        }
        if (params.omni_runtime_profile.model_explicit) {
            std::ifstream explicit_model(params.model.path, std::ios::binary);
            if (!explicit_model.good()) {
                LOG_ERR("missing explicit LLM model: %s\n", params.model.path.c_str());
                llama_backend_free();
                return 1;
            }
        }

        const auto inventory = omni::discover_model_inventory(model_root);
        const auto hardware = omni::detect_hardware_snapshot();
        omni::runtime_profile_overrides overrides;
        if (params.omni_runtime_profile.model_explicit) {
            overrides.llm_model = params.model.path;
        }
        if (params.omni_runtime_profile.n_gpu_layers_explicit) {
            overrides.n_gpu_layers = params.n_gpu_layers;
        }
        if (params.omni_runtime_profile.n_ctx_explicit) {
            overrides.n_ctx = params.n_ctx;
        }
        if (params.omni_runtime_profile.token2wav_threads_explicit) {
            overrides.token2wav_threads = params.omni_runtime_profile.token2wav_threads;
        }

        auto resolved = omni::resolve_runtime_profile(
                params.omni_runtime_profile.profile, hardware, inventory, overrides);
        if (!resolved.ok) {
            LOG_ERR("failed to resolve Omni runtime profile: %s\n", resolved.error.c_str());
            llama_backend_free();
            return 1;
        }

        runtime_config = std::move(resolved.config);
        const auto fixture = default_autotune_fixture();
        std::string cache_path;
        try {
            cache_path = params.omni_runtime_profile.autotune_cache.empty()
                             ? fs_get_cache_file("omni-runtime-autotune-v3.json")
                             : params.omni_runtime_profile.autotune_cache;
        } catch (const std::exception & exception) {
            if (!params.omni_runtime_profile.autotune.empty()) {
                LOG_ERR("autotune calibration failed: cannot resolve cache path: %s\n", exception.what());
                llama_backend_free();
                return 1;
            }
            runtime_config->autotune_cache_status = "invalid";
            runtime_config->autotune_cache_reason = std::string("cannot resolve cache path: ") + exception.what();
        }

        if (!params.omni_runtime_profile.autotune.empty()) {
            const int result = run_autotune_refresh(
                params, *runtime_config, hardware, inventory, overrides, fixture, cache_path);
            llama_backend_free();
            return result;
        }

        if (!cache_path.empty()) {
            std::error_code filesystem_error;
            const bool cache_exists = std::filesystem::exists(cache_path, filesystem_error);
            if (filesystem_error) {
                runtime_config->autotune_cache_status = "invalid";
                runtime_config->autotune_cache_reason = "cannot inspect autotune cache: " +
                                                        filesystem_error.message();
            } else if (!cache_exists) {
                runtime_config->autotune_cache_status = "missing";
                runtime_config->autotune_cache_reason = "autotune cache does not exist at " + cache_path;
            } else if (!std::filesystem::is_regular_file(cache_path, filesystem_error)) {
                runtime_config->autotune_cache_status = "invalid";
                runtime_config->autotune_cache_reason = filesystem_error
                                                            ? "cannot inspect autotune cache: " +
                                                                  filesystem_error.message()
                                                            : "autotune cache path is not a regular file";
            } else {
                std::string fingerprint_error;
                const auto fingerprint = build_autotune_fingerprint(
                    params, hardware, *runtime_config, inventory, overrides, fixture, fingerprint_error);
                if (!fingerprint_error.empty()) {
                    runtime_config->autotune_cache_status = "invalid";
                    runtime_config->autotune_cache_reason = fingerprint_error;
                } else {
                    auto cache = omni::read_autotune_cache(cache_path, fingerprint);
                    if (cache.status == omni::autotune_cache_status::applied) {
                        omni::autotune_plan_options plan_options;
                        plan_options.inventory            = inventory;
                        plan_options.model_explicit       = params.omni_runtime_profile.model_explicit;
                        plan_options.allow_quantized      = params.omni_runtime_profile.autotune_allow_quantized;
                        plan_options.hardware_concurrency = std::thread::hardware_concurrency();
                        plan_options.previous_measured_module_bytes = cache.record.memory_estimate_inputs;
                        const auto plan = omni::create_autotune_candidate_plan(
                            *runtime_config, hardware, overrides, plan_options);
                        std::string validation_error;
                        if (!omni::validate_autotune_cache_against_plan(cache.record, plan, validation_error)) {
                            cache.status = omni::autotune_cache_status::invalid;
                            cache.reason = validation_error;
                        }
                    }
                    apply_cached_autotune_config(*runtime_config, cache, hardware);
                }
            }
        }

        std::string apply_error;
        if (!omni::apply_effective_runtime_config(params, *runtime_config, apply_error)) {
            LOG_ERR("%s\n", apply_error.c_str());
            llama_backend_free();
            return 1;
        }

        const std::string effective_config = omni::format_effective_runtime_config(*runtime_config);
        if (params.omni_runtime_profile.print_effective_config) {
            std::fputs(effective_config.c_str(), stdout);
            std::fflush(stdout);
            llama_backend_free();
            return 0;
        }
        LOG_INF("Resolved Omni runtime configuration:\n%s", effective_config.c_str());
    }

    LOG_INF("Omni HTTP server starting...\n");

    // auto-detect omni model paths
    if (!params.vpm_model.empty() || !params.apm_model.empty() || !params.tts_model.empty()) {
        LOG_INF("Using explicit omni model paths from args\n");
    }

    // HTTP server setup
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    const bool has_ssl_cert = !params.ssl_file_cert.empty();
    const bool has_ssl_key  = !params.ssl_file_key.empty();
    if (has_ssl_cert != has_ssl_key) {
        LOG_ERR("--ssl-cert-file and --ssl-key-file must be provided together\n");
        llama_backend_free();
        return 1;
    }

    std::unique_ptr<httplib::Server> server;
    if (has_ssl_cert) {
        server = std::make_unique<httplib::SSLServer>(
            params.ssl_file_cert.c_str(), params.ssl_file_key.c_str());
    } else {
        server = std::make_unique<httplib::Server>();
    }
    httplib::Server & svr = *server;
#else
    httplib::Server svr;
#endif

    if (!svr.is_valid()) {
        LOG_ERR("failed to initialize the HTTP server TLS context\n");
        llama_backend_free();
        return 1;
    }

    omni_server_state state;

    // GET /health
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "comni"}};
        res.set_header("X-Engine", "comni");
        res_ok(res, health);
    });

    svr.Get("/v1/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "comni"}};
        res.set_header("X-Engine", "comni");
        res_ok(res, health);
    });

    // POST /v1/stream/omni_init
    svr.Post("/v1/stream/omni_init", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        if (!data.contains("msg_type") && !data.contains("media_type")) {
            res_error(res, format_error_response("\"msg_type\" or \"media_type\" must be provided"));
            return;
        }

        int media_type = data.value("msg_type", data.value("media_type", 2));
        bool use_tts   = data.value("use_tts", true);
        const auto session_options = omni::resolve_runtime_session_options(
            runtime_config ? &*runtime_config : nullptr,
            data.value("duplex_mode", false),
            data.value("tts_gpu_layers", 100),
            data.value("token2wav_device", "gpu:0"),
            data.value("token2wav_threads", params.omni_runtime_profile.token2wav_threads));
        bool duplex_mode = session_options.duplex_mode;
        int tts_gpu_layers = session_options.tts_gpu_layers;
        std::string token2wav_device = session_options.token2wav_device;
        int token2wav_threads = session_options.token2wav_threads;
        std::string output_dir = data.value("output_dir", "./tools/omni/output");
        std::string voice_audio = data.value("voice_audio", "");
        std::string tts_ref_audio = data.value("tts_ref_audio", voice_audio);

        // validate key files
        auto check_file = [&](const std::string & role, const std::string & path) -> bool {
            if (path.empty()) return true;
            std::ifstream f(path);
            if (!f.good()) {
                res_error(res, format_error_response(
                    "omni_init missing required model file (" + role + "): " + path));
                return false;
            }
            return true;
        };

        // Keep legacy HTTP aligned with /backend: the LLM path (-m) anchors the
        // fixed MiniCPM-o sub-model layout; request model_dir is ignored.
        if (!ensure_omni_model_paths_from_llm(params)) {
            res_error(res, format_error_response("LLM model path (-m) is required to derive omni model paths"));
            return;
        }

        if (!check_file("LLM",    params.model.path) ||
            !check_file("vision", params.vpm_model)  ||
            !check_file("audio",  params.apm_model)  ||
            (use_tts && !check_file("tts", params.tts_model))) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx) {
                omni_free(state.octx);
                state.octx = nullptr;
            }
        }

        omni_context * octx = omni_init(&params, media_type, use_tts, params.tts_bin_dir, tts_gpu_layers,
                                         token2wav_device, duplex_mode,
                                         /*existing_model=*/nullptr, /*existing_ctx=*/nullptr, output_dir,
                                         runtime_config ? &*runtime_config : nullptr,
                                         session_options.strict_runtime_config, token2wav_threads);
        if (!octx) {
            res_error(res, format_error_response("omni_init failed"));
            return;
        }

        // voice clone / assistant prompt
        if (data.contains("voice_clone_prompt")) octx->omni_voice_clone_prompt = data["voice_clone_prompt"];
        if (data.contains("assistant_prompt")) octx->omni_assistant_prompt = data["assistant_prompt"];
        octx->ref_audio_path = voice_audio;
        if (use_tts && !tts_ref_audio.empty()) {
            if (!omni_set_tts_reference_audio(octx, tts_ref_audio)) {
                omni_free(octx);
                res_error(res, format_error_response("tts_voice_prepare_failed"));
                return;
            }
        }

        // The HTTP contract reserves cnt=0 for session initialization. Complete
        // that prefill here so the first client frame can start at cnt=1.
        if (!stream_prefill(octx, voice_audio, /*img_fname=*/"", /*index=*/0)) {
            omni_free(octx);
            res_error(res, format_error_response("omni_init failed to initialize the streaming session"));
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            state.octx = octx;
        }

        res_ok(res, {{"success", true}});
    });

    // POST /v1/stream/prefill
    svr.Post("/v1/stream/prefill", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        if (!data.contains("audio_path_prefix") || !data.at("audio_path_prefix").is_string()) {
            res_error(res, format_error_response("\"audio_path_prefix\" must be provided as string"));
            return;
        }
        if (!data.contains("cnt") || !data.at("cnt").is_number_integer()) {
            res_error(res, format_error_response("\"cnt\" must be provided as integer"));
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx == nullptr) {
                res_error(res, format_error_response("omni context not initialized. call /v1/stream/omni_init first"));
                return;
            }
        }

        std::string audio_path = data.at("audio_path_prefix");
        std::string img_path   = data.value("img_path_prefix", "");
        std::string text       = data.value("text", "");
        int cnt                = data.at("cnt");
        int max_slice_nums     = data.value("max_slice_nums", -1);

        bool ok = false;
        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            ok = stream_prefill(state.octx, audio_path, img_path, cnt, max_slice_nums, text);
        }

        if (!ok) {
            res_error(res, format_error_response("stream_prefill failed"));
            return;
        }

        res_ok(res, {{"success", true}, {"audio_path_prefix", audio_path}, {"cnt", cnt}});
    });

    // POST /v1/stream/decode (SSE)
    svr.Post("/v1/stream/decode", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx == nullptr) {
                res_error(res, format_error_response("omni context not initialized. call /v1/stream/omni_init first"));
                return;
            }
        }

        std::string debug_dir = data.value("debug_dir", "./");
        bool stream = data.value("stream", true);
        int round_idx = data.value("round_idx", -1);

        // length_penalty
        if (data.contains("length_penalty") && data.at("length_penalty").is_number()) {
            float lp = data.at("length_penalty").get<float>();
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx != nullptr) {
                state.octx->length_penalty = lp;
            }
        }

        if (!stream) {
            bool ok = false;
            {
                std::lock_guard<std::mutex> lock(state.octx_mutex);
                ok = stream_decode(state.octx, debug_dir, round_idx);
            }
            if (!ok) {
                res_error(res, format_error_response("stream_decode failed"));
                return;
            }
            res_ok(res, {{"success", true}});
            return;
        }

        // SSE streaming
        res.set_chunked_content_provider("text/event-stream",
            [&](size_t, httplib::DataSink & sink) -> bool {
                // reset state
                {
                    std::lock_guard<std::mutex> lock(state.octx->text_mtx);
                    state.octx->text_queue.clear();
                    state.octx->text_done_flag = false;
                    state.octx->text_streaming = true;
                }

                // start decode in background thread
                std::thread worker([&](std::string dd, int ri) {
                    std::lock_guard<std::mutex> lock(state.octx_mutex);
                    (void) stream_decode(state.octx, dd, ri);
                }, debug_dir, round_idx);

                // poll text queue
                while (true) {
                    std::unique_lock<std::mutex> lk(state.octx->text_mtx);
                    state.octx->text_cv.wait_for(lk, std::chrono::milliseconds(200), [&]{
                        return !state.octx->text_queue.empty() || state.octx->text_done_flag;
                    });

                    while (!state.octx->text_queue.empty()) {
                        std::string frag = std::move(state.octx->text_queue.front());
                        state.octx->text_queue.pop_front();
                        lk.unlock();

                        json ev;
                        if (frag == "__IS_LISTEN__") {
                            ev = {{"content", ""}, {"stop", false}, {"is_listen", true}, {"end_of_turn", true}};
                        } else if (frag == "__END_OF_TURN__") {
                            ev = {{"content", ""}, {"stop", true}, {"is_listen", false}, {"end_of_turn", true}};
                        } else {
                            ev = {{"content", frag}, {"stop", false}, {"is_listen", false}, {"end_of_turn", false}};
                        }

                        if (!server_sent_event(sink, ev)) {
                            if (worker.joinable()) worker.join();
                            return false;
                        }
                        lk.lock();
                    }

                    if (state.octx->text_done_flag) break;
                }

                if (worker.joinable()) worker.join();

                // send done
                static const std::string ev_done = "data: [DONE]\n\n";
                sink.write(ev_done.data(), ev_done.size());
                // Mark the chunked response complete. Returning false here
                // skips cpp-httplib's terminating chunk and makes clients
                // report an incomplete transfer.
                sink.done();
                return true;
            });
    });

    // POST /v1/stream/update_session_config
    svr.Post("/v1/stream/update_session_config", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        int media_type = data.value("media_type", -1);

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx == nullptr) {
                res_error(res, format_error_response("omni context not initialized"));
                return;
            }
            if (media_type > 0) {
                state.octx->media_type = media_type;
            }
        }

        res_ok(res, {{"success", true}});
    });

    //
    // Backend Protocol (WebSocket + HTTP unary)
    //
    svr.WebSocket("/backend", [&](const httplib::Request &, httplib::ws::WebSocket & ws) {
        handle_ws_backend(ws, state.session_mgr, params,
                          /*model*/nullptr, /*ctx*/nullptr,
                          state.octx, state.octx_mutex,
                          runtime_config ? &*runtime_config : nullptr);
    });

    svr.Post("/sessions/:session_id/close", [&](const httplib::Request & req, httplib::Response & res) {
        std::string session_id = req.path_params.at("session_id");
        LOG_INF("Close session requested: %s\n", session_id.c_str());

        auto * session = state.session_mgr.get(session_id);
        if (!session || session->state != SessionState::ACTIVE) {
            res_error(res, format_error_response("session not found", "not_found"));
            res.status = 404;
            return;
        }

        state.session_mgr.request_transport_close(session_id);

        // close is a completion primitive: do not return until inference
        // threads are stopped and the shared omni_context is safe to reuse.
        {
            std::lock_guard<std::mutex> octx_lock(state.octx_mutex);
            auto * closing = state.session_mgr.get(session_id);
            if (closing && closing->octx) {
                closing->octx->break_event = true;
                {
                    std::lock_guard<std::mutex> lk(closing->octx->text_mtx);
                    closing->octx->text_queue.clear();
                    closing->octx->text_done_flag = true;
                }
                closing->octx->text_cv.notify_all();
                omni_prepare_for_reuse(closing->octx);
            }

            state.session_mgr.close(session_id);
        }

        json resp;
        resp["ok"] = true;
        resp["session_id"] = session_id;
        resp["closed"] = true;
        res_ok(res, resp);
    });

    // start server
    if (!svr.listen(params.hostname, params.port)) {
        LOG_ERR("failed to bind the Omni HTTP server to %s:%d\n",
                params.hostname.c_str(), params.port);
        llama_backend_free();
        return 1;
    }

    // cleanup
    {
        std::lock_guard<std::mutex> lock(state.octx_mutex);
        if (state.octx) {
            omni_free(state.octx);
            state.octx = nullptr;
        }
    }
    llama_backend_free();

    return 0;
}
