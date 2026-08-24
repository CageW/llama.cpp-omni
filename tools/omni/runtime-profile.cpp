#include "runtime-profile.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>

namespace omni {

static constexpr uint64_t GiB = 1024ULL * 1024ULL * 1024ULL;

static std::string existing_file(const std::filesystem::path & root, const std::filesystem::path & relative_path) {
    const auto      path = root / relative_path;
    std::error_code error;
    if (std::filesystem::is_regular_file(path, error)) {
        return path.string();
    }
    return {};
}

model_inventory discover_model_inventory(const std::string & model_root) {
    model_inventory inventory;
    inventory.root = model_root;
    const std::filesystem::path root(model_root);

    inventory.llm_f16    = existing_file(root, "MiniCPM-o-4_5-F16.gguf");
    inventory.llm_q8_0   = existing_file(root, "MiniCPM-o-4_5-Q8_0.gguf");
    inventory.llm_q4_k_m = existing_file(root, "MiniCPM-o-4_5-Q4_K_M.gguf");
    inventory.vision     = existing_file(root, "vision/MiniCPM-o-4_5-vision-F16.gguf");
    inventory.audio      = existing_file(root, "audio/MiniCPM-o-4_5-audio-F16.gguf");
    inventory.tts        = existing_file(root, "tts/MiniCPM-o-4_5-tts-F16.gguf");
    inventory.projector  = existing_file(root, "tts/MiniCPM-o-4_5-projector-F16.gguf");

    const auto      token2wav_dir = root / "token2wav-gguf";
    std::error_code error;
    if (std::filesystem::is_directory(token2wav_dir, error)) {
        inventory.token2wav_dir = token2wav_dir.string();
    }
    inventory.token2wav_encoder       = existing_file(root, "token2wav-gguf/encoder.gguf");
    inventory.token2wav_flow_matching = existing_file(root, "token2wav-gguf/flow_matching.gguf");
    inventory.token2wav_flow_extra    = existing_file(root, "token2wav-gguf/flow_extra.gguf");
    inventory.token2wav_hifigan       = existing_file(root, "token2wav-gguf/hifigan2.gguf");
    inventory.token2wav_prompt_cache  = existing_file(root, "token2wav-gguf/prompt_cache.gguf");
    return inventory;
}

hardware_snapshot detect_hardware_snapshot() {
    hardware_snapshot snapshot;
    ggml_backend_load_all();
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == nullptr) {
            continue;
        }

        size_t free_memory  = 0;
        size_t total_memory = 0;
        ggml_backend_dev_memory(device, &free_memory, &total_memory);
        const auto device_type = ggml_backend_dev_type(device);
        const char * type_name = device_type == GGML_BACKEND_DEVICE_TYPE_CPU ? "CPU" :
                                 device_type == GGML_BACKEND_DEVICE_TYPE_GPU ? "GPU" : "ACCEL";
        snapshot.devices.push_back({
            ggml_backend_dev_name(device),
            ggml_backend_dev_description(device),
            device_type != GGML_BACKEND_DEVICE_TYPE_CPU,
            static_cast<uint64_t>(free_memory),
            static_cast<uint64_t>(total_memory),
            type_name,
        });
    }
    return snapshot;
}

static std::string memory_rejection_reason(const std::string & quantization, uint64_t required, uint64_t available) {
    std::ostringstream message;
    message << quantization << " rejected: estimated " << required / GiB << " GiB exceeds " << available / GiB
            << " GiB free memory";
    return message.str();
}

static std::vector<const runtime_device *> accelerator_devices(const hardware_snapshot & hardware) {
    std::vector<const runtime_device *> devices;
    for (const auto & device : hardware.devices) {
        if (device.is_accelerator) {
            devices.push_back(&device);
        }
    }
    std::sort(devices.begin(), devices.end(), [](const runtime_device * lhs, const runtime_device * rhs) {
        if (lhs->free_memory != rhs->free_memory) {
            return lhs->free_memory > rhs->free_memory;
        }
        return lhs->name < rhs->name;
    });
    return devices;
}

static std::string token2wav_device_for(const runtime_device & device) {
    std::string digits;
    for (auto it = device.name.rbegin(); it != device.name.rend() && std::isdigit(static_cast<unsigned char>(*it)); ++it) {
        digits.push_back(*it);
    }
    if (!digits.empty()) {
        std::reverse(digits.begin(), digits.end());
        return "gpu:" + digits;
    }
    return "gpu:0";
}

static const runtime_device * find_device(const hardware_snapshot & hardware, const std::string & name) {
    for (const auto & device : hardware.devices) {
        if (device.name == name) {
            return &device;
        }
    }
    return nullptr;
}

static std::string normalize_device(const std::string & requested,
                                    const hardware_snapshot & hardware,
                                    const runtime_device * primary,
                                    const std::string & module,
                                    std::string & error) {
    if (requested.empty() || requested == "cpu") {
        return requested.empty() ? "cpu" : requested;
    }
    if (requested == "gpu") {
        if (primary != nullptr) {
            return primary->name;
        }
        error = "unsupported " + module + " placement device: gpu (no accelerator is available)";
        return {};
    }
    if (requested.rfind("gpu:", 0) == 0) {
        try {
            const size_t index = static_cast<size_t>(std::stoul(requested.substr(4)));
            const auto accelerators = accelerator_devices(hardware);
            if (index < accelerators.size()) {
                return accelerators[index]->name;
            }
        } catch (...) {
        }
    }
    if (find_device(hardware, requested) != nullptr) {
        return requested;
    }
    error = "unsupported " + module + " placement device: " + requested;
    return {};
}

static void add_module_placement(effective_runtime_config & config,
                                 const std::string & module,
                                 const std::string & model,
                                 const std::string & precision,
                                 const std::string & device,
                                 const std::string & execution = "independent") {
    config.placements.push_back({ module, model, precision, device,
                                  device == "cpu" ? "cpu" : "gpu", execution, "active" });
}

static void apply_accelerator_defaults(effective_runtime_config & config, const runtime_device & primary) {
    config.primary_device_name         = primary.name;
    config.primary_device_description  = primary.description;
    config.primary_device_free_memory  = primary.free_memory;
    config.primary_device_total_memory = primary.total_memory;
    config.llm_device                  = primary.name;
    config.n_gpu_layers                = -2;
    config.tts_gpu_layers              = -1;
    config.token2wav_device            = token2wav_device_for(primary);
    config.n_ctx                       = 8192;
    config.duplex_mode                 = true;
    config.async_mode                  = true;
    config.vpm_batch_encode            = true;
}

static std::string expected_path(const model_inventory & inventory, const char * relative_path) {
    if (inventory.root.empty()) {
        return relative_path;
    }
    return inventory.root + "/" + relative_path;
}

static std::string validate_required_modules(const model_inventory & inventory) {
    struct required_module {
        const char *        name;
        const std::string * path;
        const char *        relative_path;
    };

    const required_module modules[] = {
        { "vision",                  &inventory.vision,                  "vision/MiniCPM-o-4_5-vision-F16.gguf" },
        { "audio",                   &inventory.audio,                   "audio/MiniCPM-o-4_5-audio-F16.gguf"   },
        { "tts",                     &inventory.tts,                     "tts/MiniCPM-o-4_5-tts-F16.gguf"       },
        { "projector",               &inventory.projector,               "tts/MiniCPM-o-4_5-projector-F16.gguf" },
        { "token2wav encoder",       &inventory.token2wav_encoder,       "token2wav-gguf/encoder.gguf"          },
        { "token2wav flow_matching", &inventory.token2wav_flow_matching, "token2wav-gguf/flow_matching.gguf"    },
        { "token2wav flow_extra",    &inventory.token2wav_flow_extra,    "token2wav-gguf/flow_extra.gguf"       },
        { "token2wav hifigan2",      &inventory.token2wav_hifigan,       "token2wav-gguf/hifigan2.gguf"         },
        { "token2wav prompt_cache",  &inventory.token2wav_prompt_cache,  "token2wav-gguf/prompt_cache.gguf"     },
    };

    for (const auto & module : modules) {
        if (module.path->empty()) {
            return std::string("missing required ") + module.name +
                   " model: " + expected_path(inventory, module.relative_path);
        }
    }
    return {};
}

static void apply_cpu_defaults(effective_runtime_config & config) {
    config.llm_device       = "cpu";
    config.n_gpu_layers     = 0;
    config.tts_gpu_layers   = 0;
    config.token2wav_device = "cpu";
    config.n_ctx            = 4096;
    config.duplex_mode      = false;
    config.async_mode       = true;
    config.vpm_batch_encode = false;
}

static bool select_cpu_model(const model_inventory & inventory, effective_runtime_config & config) {
    if (!inventory.llm_q4_k_m.empty()) {
        config.resolved_profile = "full_omni_cpu_q4_k_m";
        config.llm_model        = inventory.llm_q4_k_m;
        config.llm_quantization = "Q4_K_M";
    } else if (!inventory.llm_q8_0.empty()) {
        config.resolved_profile = "full_omni_cpu_q8_0";
        config.llm_model        = inventory.llm_q8_0;
        config.llm_quantization = "Q8_0";
    } else if (!inventory.llm_f16.empty()) {
        config.resolved_profile = "full_omni_cpu_f16";
        config.llm_model        = inventory.llm_f16;
        config.llm_quantization = "F16";
    } else {
        return false;
    }

    config.reason = "no accelerator backend was detected; selected the smallest available CPU-compatible model";
    apply_cpu_defaults(config);
    return true;
}

static std::string quantization_from_path(const std::string & path) {
    std::string upper = path;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (upper.find("Q4_K_M") != std::string::npos) {
        return "Q4_K_M";
    }
    if (upper.find("Q8_0") != std::string::npos) {
        return "Q8_0";
    }
    if (upper.find("F16") != std::string::npos) {
        return "F16";
    }
    return "UNKNOWN";
}

static bool apply_module_inventory(effective_runtime_config & config,
                                   const model_inventory & inventory,
                                   const hardware_snapshot & hardware,
                                   const runtime_device * primary,
                                   const runtime_device * secondary,
                                   const runtime_profile_overrides & overrides,
                                   std::string & error) {
    config.vision_model        = inventory.vision;
    config.audio_model         = inventory.audio;
    config.tts_model           = inventory.tts;
    config.projector_model     = inventory.projector;
    config.token2wav_model_dir = inventory.token2wav_dir;

    const std::string primary_device = primary ? primary->name : "cpu";
    const std::string secondary_device = secondary ? secondary->name : primary_device;
    const bool split_modules = primary != nullptr && secondary != nullptr;

    auto choose_device = [&](const std::string & module,
                             const std::string & default_device,
                             const std::optional<std::string> & override) -> std::string {
        return normalize_device(override ? *override : default_device, hardware, primary, module, error);
    };

    config.llm_device = choose_device("llm", primary_device, overrides.llm_device);
    config.vision_device = choose_device("vision", split_modules ? secondary_device : primary_device,
                                         overrides.vision_device);
    config.audio_device = choose_device("audio", split_modules ? secondary_device : primary_device,
                                        overrides.audio_device);
    config.tts_device = choose_device("tts", primary_device, overrides.tts_device);
    const std::string projector_device = choose_device("projector", split_modules ? secondary_device : primary_device,
                                                       overrides.projector_device);
    const std::string token2wav_backend_device = choose_device("token2wav", primary_device,
                                                               overrides.token2wav_device);
    if (!error.empty()) {
        return false;
    }

    if (config.llm_device == "cpu") {
        config.n_gpu_layers = 0;
    }
    if (config.tts_device == "cpu") {
        config.tts_gpu_layers = 0;
    }

    config.token2wav_device = token2wav_backend_device == "cpu" ? "cpu" :
                              (overrides.token2wav_device && overrides.token2wav_device->rfind("gpu:", 0) == 0
                                   ? *overrides.token2wav_device
                                   : token2wav_device_for(*find_device(hardware, token2wav_backend_device)));

    config.placements.clear();
    add_module_placement(config, "llm", config.llm_model, config.llm_quantization, config.llm_device);
    add_module_placement(config, "vision", config.vision_model, "F16", config.vision_device);
    add_module_placement(config, "audio", config.audio_model, "F16", config.audio_device);
    add_module_placement(config, "tts", config.tts_model, "F16", config.tts_device);
    add_module_placement(config, "projector", config.projector_model, "F16", projector_device);
    add_module_placement(config, "token2wav", config.token2wav_model_dir, "F16", token2wav_backend_device);
    return true;
}

static void apply_overrides(effective_runtime_config &        config,
                            const runtime_profile_overrides & overrides,
                            bool                              has_accelerator) {
    if (overrides.llm_model) {
        config.llm_model        = *overrides.llm_model;
        config.llm_quantization = quantization_from_path(*overrides.llm_model);
        if (has_accelerator) {
            const bool multi_accelerator = config.resolved_profile.rfind("full_omni_multi_accelerator_", 0) == 0;
            if (config.llm_quantization == "Q4_K_M") {
                config.resolved_profile = multi_accelerator ? "full_omni_multi_accelerator_q4_k_m"
                                                            : "full_omni_accelerator_q4_k_m";
            } else if (config.llm_quantization == "Q8_0") {
                config.resolved_profile = multi_accelerator ? "full_omni_multi_accelerator_q8_0"
                                                            : "full_omni_accelerator_q8_0";
            } else if (config.llm_quantization == "F16") {
                config.resolved_profile = multi_accelerator ? "full_omni_multi_accelerator_f16"
                                                            : "full_omni_accelerator_f16";
            } else {
                config.resolved_profile = multi_accelerator ? "full_omni_multi_accelerator_explicit"
                                                            : "full_omni_accelerator_explicit";
            }
        }
        config.reason += "; explicit LLM model override preserved";
    }
    if (overrides.n_gpu_layers) {
        config.n_gpu_layers = *overrides.n_gpu_layers;
    }
    if (overrides.n_ctx) {
        config.n_ctx = *overrides.n_ctx;
    }
    if (overrides.token2wav_threads) {
        config.token2wav_threads = *overrides.token2wav_threads;
    }
    if (overrides.vpm_batch_encode) {
        config.vpm_batch_encode = *overrides.vpm_batch_encode;
    }
    for (auto & placement : config.placements) {
        if (placement.module == "llm") {
            placement.model = config.llm_model;
            placement.precision = config.llm_quantization;
            placement.device = config.llm_device;
            break;
        }
    }
}

runtime_profile_result resolve_runtime_profile(const std::string &               requested_profile,
                                               const hardware_snapshot &         hardware,
                                               const model_inventory &           inventory,
                                               const runtime_profile_overrides & overrides) {
    runtime_profile_result result;
    result.config.requested_profile = requested_profile;

    const auto accelerators = accelerator_devices(hardware);
    const runtime_device * primary = accelerators.empty() ? nullptr : accelerators.front();
    const runtime_device * secondary = accelerators.size() > 1 ? accelerators[1] : nullptr;

    if (requested_profile != "auto") {
        result.error = "unsupported runtime profile: " + requested_profile;
        return result;
    }

    result.error = validate_required_modules(inventory);
    if (!result.error.empty()) {
        return result;
    }

    const bool has_accelerator = primary != nullptr;
    const bool split_modules = secondary != nullptr && secondary->free_memory >= 4 * GiB;
    if (!split_modules) {
        secondary = nullptr;
    }
    if (overrides.llm_model) {
        result.ok                      = true;
        result.config.resolved_profile = has_accelerator
                                             ? (split_modules ? "full_omni_multi_accelerator_explicit"
                                                              : "full_omni_accelerator_explicit")
                                             : "full_omni_cpu_explicit";
        result.config.reason           = "explicit LLM model was provided; profile selected execution defaults only";
        result.config.llm_model        = *overrides.llm_model;
        result.config.llm_quantization = quantization_from_path(*overrides.llm_model);
        if (has_accelerator) {
            apply_accelerator_defaults(result.config, *primary);
        } else {
            apply_cpu_defaults(result.config);
        }
        if (!apply_module_inventory(result.config, inventory, hardware, primary, secondary, overrides, result.error)) {
            result.ok = false;
            return result;
        }
        apply_overrides(result.config, overrides, has_accelerator);
        return result;
    }

    if (has_accelerator) {
        struct candidate {
            const char *        quantization;
            const char *        resolved_profile;
            const std::string * model;
            uint64_t            required_memory;
        };

        const candidate candidates[] = {
            { "F16",    "full_omni_accelerator_f16",    &inventory.llm_f16,    20 * GiB },
            { "Q8_0",   "full_omni_accelerator_q8_0",   &inventory.llm_q8_0,   13 * GiB },
            { "Q4_K_M", "full_omni_accelerator_q4_k_m", &inventory.llm_q4_k_m, 9 * GiB  },
        };

        for (const auto & candidate : candidates) {
            if (candidate.model->empty()) {
                result.config.fallback_reasons.push_back(std::string(candidate.quantization) +
                                                         " rejected: model file is not available");
                continue;
            }
            if (primary->free_memory < candidate.required_memory) {
                result.config.fallback_reasons.push_back(
                    memory_rejection_reason(candidate.quantization, candidate.required_memory, primary->free_memory));
                continue;
            }

            result.ok                      = true;
            result.config.resolved_profile = split_modules
                                                 ? std::string("full_omni_multi_accelerator_") +
                                                       (candidate.quantization == std::string("F16") ? "f16" :
                                                        candidate.quantization == std::string("Q8_0") ? "q8_0" : "q4_k_m")
                                                 : candidate.resolved_profile;
            result.config.reason    = "primary accelerator has enough free memory for the selected full Omni estimate";
            result.config.llm_model = *candidate.model;
            result.config.llm_quantization          = candidate.quantization;
            result.config.estimated_required_memory = candidate.required_memory;
            apply_accelerator_defaults(result.config, *primary);
            if (!apply_module_inventory(result.config, inventory, hardware, primary, secondary, overrides, result.error)) {
                result.ok = false;
                return result;
            }
            apply_overrides(result.config, overrides, true);
            return result;
        }
    } else if (select_cpu_model(inventory, result.config)) {
        result.ok = true;
        if (!apply_module_inventory(result.config, inventory, hardware, nullptr, nullptr, overrides, result.error)) {
            result.ok = false;
            return result;
        }
        apply_overrides(result.config, overrides, false);
        return result;
    }

    result.error = has_accelerator ? "no matching runtime profile" :
                                     "no LLM model in F16, Q8_0, or Q4_K_M is available for runtime profile auto";
    return result;
}

std::string format_effective_runtime_config(const effective_runtime_config & config) {
    std::ostringstream output;
    output << "profile=" << config.requested_profile << '\n';
    output << "resolved_profile=" << config.resolved_profile << '\n';
    output << "reason=" << config.reason << '\n';
    if (!config.primary_device_name.empty()) {
        output << "primary_device=" << config.primary_device_name;
        if (!config.primary_device_description.empty()) {
            output << " (" << config.primary_device_description << ')';
        }
        output << '\n';
        output << "primary_device_free_memory_mib=" << config.primary_device_free_memory / (1024ULL * 1024ULL) << '\n';
        output << "primary_device_total_memory_mib=" << config.primary_device_total_memory / (1024ULL * 1024ULL)
               << '\n';
    }
    if (config.estimated_required_memory > 0) {
        output << "estimated_required_memory_mib=" << config.estimated_required_memory / (1024ULL * 1024ULL) << '\n';
    }
    output << "llm_model=" << config.llm_model << '\n';
    output << "llm_quantization=" << config.llm_quantization << '\n';
    output << "llm_device=" << config.llm_device << '\n';
    output << "llm_n_gpu_layers=" << config.n_gpu_layers << '\n';
    output << "vision_model=" << config.vision_model << '\n';
    output << "vision_device=" << config.vision_device << '\n';
    output << "audio_model=" << config.audio_model << '\n';
    output << "audio_device=" << config.audio_device << '\n';
    output << "tts_model=" << config.tts_model << '\n';
    output << "tts_device=" << config.tts_device << '\n';
    output << "tts_gpu_layers=" << config.tts_gpu_layers << '\n';
    output << "projector_model=" << config.projector_model << '\n';
    output << "token2wav_model_dir=" << config.token2wav_model_dir << '\n';
    output << "token2wav_device=" << config.token2wav_device << '\n';
    output << "token2wav_threads=" << config.token2wav_threads << '\n';
    output << "execution=" << (config.async_mode ? "async" : "sync") << ','
           << (config.duplex_mode ? "duplex" : "simplex");
    if (config.vpm_batch_encode) {
        output << ",vpm_batch_encode";
    }
    output << ",ctx_size=" << config.n_ctx << '\n';
    if (!config.autotune_cache_status.empty()) {
        output << "autotune_cache=" << config.autotune_cache_status << '\n';
        output << "autotune_cache_reason=" << config.autotune_cache_reason << '\n';
        if (!config.autotune_selected_candidate.empty()) {
            output << "autotune_selected_candidate=" << config.autotune_selected_candidate << '\n';
            output << "output_consistent_with_baseline="
                   << (config.output_consistent_with_baseline ? "true" : "false") << '\n';
        }
        output << "audio_semantics_verified=" << (config.audio_semantics_verified ? "true" : "false") << '\n';
    }
    output << "placement_plan_count=" << config.placements.size() << '\n';
    for (const auto & placement : config.placements) {
        output << "placement=" << placement.module
               << ",model=" << placement.model
               << ",precision=" << placement.precision
               << ",device=" << placement.device
               << ",backend=" << placement.backend
               << ",execution=" << placement.execution
               << ",status=" << placement.status << '\n';
    }
    for (const auto & fallback : config.fallback_reasons) {
        output << "fallback=" << fallback << '\n';
    }
    return output.str();
}

}  // namespace omni
