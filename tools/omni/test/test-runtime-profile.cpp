#include "arg.h"
#include "common.h"
#include "runtime-profile.h"

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static constexpr uint64_t GiB = 1024ULL * 1024ULL * 1024ULL;

class temporary_model_tree {
  public:
    temporary_model_tree() {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() / ("omni-runtime-profile-test-" + std::to_string(unique));
        std::filesystem::create_directories(root);
    }

    ~temporary_model_tree() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    void touch(const std::filesystem::path & relative_path) const {
        const auto path = root / relative_path;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        file << "test";
    }

    std::filesystem::path root;
};

static omni::model_inventory complete_inventory() {
    omni::model_inventory inventory;
    inventory.root                    = "/models/MiniCPM-o-4_5-gguf";
    inventory.llm_f16                 = inventory.root + "/MiniCPM-o-4_5-F16.gguf";
    inventory.llm_q8_0                = inventory.root + "/MiniCPM-o-4_5-Q8_0.gguf";
    inventory.llm_q4_k_m              = inventory.root + "/MiniCPM-o-4_5-Q4_K_M.gguf";
    inventory.vision                  = inventory.root + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    inventory.audio                   = inventory.root + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    inventory.tts                     = inventory.root + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    inventory.projector               = inventory.root + "/tts/MiniCPM-o-4_5-projector-F16.gguf";
    inventory.token2wav_dir           = inventory.root + "/token2wav-gguf";
    inventory.token2wav_encoder       = inventory.token2wav_dir + "/encoder.gguf";
    inventory.token2wav_flow_matching = inventory.token2wav_dir + "/flow_matching.gguf";
    inventory.token2wav_flow_extra    = inventory.token2wav_dir + "/flow_extra.gguf";
    inventory.token2wav_hifigan       = inventory.token2wav_dir + "/hifigan2.gguf";
    inventory.token2wav_prompt_cache  = inventory.token2wav_dir + "/prompt_cache.gguf";
    return inventory;
}

static void test_auto_selects_f16_when_accelerator_memory_is_sufficient() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });

    const auto result = omni::resolve_runtime_profile("auto", hardware, complete_inventory());

    assert(result.ok);
    assert(result.config.resolved_profile == "full_omni_accelerator_f16");
    assert(result.config.llm_model == "/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf");
    assert(result.config.llm_quantization == "F16");
    assert(result.config.n_gpu_layers == -2);
    assert(result.config.tts_gpu_layers == -1);
    assert(result.config.token2wav_device == "gpu:0");
    assert(result.config.n_ctx == 8192);
    assert(result.config.duplex_mode);
    assert(result.config.async_mode);
    assert(result.config.vpm_batch_encode);
}

static void test_auto_falls_back_to_q8_when_f16_does_not_fit() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 15 * GiB, 96 * GiB, "GPU" });

    const auto result = omni::resolve_runtime_profile("auto", hardware, complete_inventory());

    assert(result.ok);
    assert(result.config.resolved_profile == "full_omni_accelerator_q8_0");
    assert(result.config.llm_quantization == "Q8_0");
    assert(result.config.llm_model == "/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q8_0.gguf");
    assert(result.config.fallback_reasons.size() == 1);
    assert(result.config.fallback_reasons.front().find("F16") != std::string::npos);
}

static void test_auto_falls_back_to_q4_when_q8_does_not_fit() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 10 * GiB, 96 * GiB, "GPU" });

    const auto result = omni::resolve_runtime_profile("auto", hardware, complete_inventory());

    assert(result.ok);
    assert(result.config.resolved_profile == "full_omni_accelerator_q4_k_m");
    assert(result.config.llm_quantization == "Q4_K_M");
    assert(result.config.llm_model == "/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf");
    assert(result.config.fallback_reasons.size() == 2);
    assert(result.config.fallback_reasons.at(0).find("F16") != std::string::npos);
    assert(result.config.fallback_reasons.at(1).find("Q8_0") != std::string::npos);
}

static void test_auto_reports_the_missing_omni_module() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    auto inventory = complete_inventory();
    inventory.vision.clear();

    const auto result = omni::resolve_runtime_profile("auto", hardware, inventory);

    assert(!result.ok);
    assert(result.error.find("vision") != std::string::npos);
    assert(result.error.find("MiniCPM-o-4_5-vision-F16.gguf") != std::string::npos);
}

static void test_auto_uses_cpu_compatible_profile_without_an_accelerator() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CPU", "Apple M-series CPU", false, 48 * GiB, 64 * GiB, "CPU" });

    const auto result = omni::resolve_runtime_profile("auto", hardware, complete_inventory());

    assert(result.ok);
    assert(result.config.resolved_profile == "full_omni_cpu_q4_k_m");
    assert(result.config.llm_quantization == "Q4_K_M");
    assert(result.config.n_gpu_layers == 0);
    assert(result.config.tts_gpu_layers == 0);
    assert(result.config.token2wav_device == "cpu");
    assert(result.config.llm_device == "cpu");
    assert(result.config.n_ctx == 4096);
    assert(!result.config.vpm_batch_encode);
}

static void test_zero_memory_accelerator_is_not_misclassified_as_cpu_only() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CPU", "Apple M-series CPU", false, 48 * GiB, 64 * GiB, "CPU" });
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 0, 96 * GiB, "GPU" });

    const auto result = omni::resolve_runtime_profile("auto", hardware, complete_inventory());

    assert(!result.ok);
    assert(result.error == "no matching runtime profile");
}

static void test_explicit_values_override_auto_recommendations() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    const auto                      inventory = complete_inventory();
    omni::runtime_profile_overrides overrides;
    overrides.llm_model        = inventory.llm_q4_k_m;
    overrides.n_gpu_layers     = 37;
    overrides.n_ctx            = 6144;
    overrides.token2wav_threads = 32;
    overrides.vpm_batch_encode = false;

    const auto result = omni::resolve_runtime_profile("auto", hardware, inventory, overrides);

    assert(result.ok);
    assert(result.config.llm_model == inventory.llm_q4_k_m);
    assert(result.config.llm_quantization == "Q4_K_M");
    assert(result.config.n_gpu_layers == 37);
    assert(result.config.n_ctx == 6144);
    assert(result.config.token2wav_threads == 32);
    assert(!result.config.vpm_batch_encode);
}

static void test_explicit_model_works_without_recommended_llm_candidates() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    auto inventory = complete_inventory();
    inventory.llm_f16.clear();
    inventory.llm_q8_0.clear();
    inventory.llm_q4_k_m.clear();
    omni::runtime_profile_overrides overrides;
    overrides.llm_model = "/models/custom-omni.gguf";

    const auto result = omni::resolve_runtime_profile("auto", hardware, inventory, overrides);

    assert(result.ok);
    assert(result.config.resolved_profile == "full_omni_accelerator_explicit");
    assert(result.config.llm_model == "/models/custom-omni.gguf");
    assert(result.config.llm_quantization == "UNKNOWN");
    assert(result.config.n_gpu_layers == -2);
}

static void test_effective_config_explains_modules_and_fallbacks() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 15 * GiB, 96 * GiB, "GPU" });

    const auto        result = omni::resolve_runtime_profile("auto", hardware, complete_inventory());
    const std::string output = omni::format_effective_runtime_config(result.config);

    assert(output.find("profile=auto\n") != std::string::npos);
    assert(output.find("resolved_profile=full_omni_accelerator_q8_0\n") != std::string::npos);
    assert(output.find("primary_device=CUDA0 (NVIDIA H20)\n") != std::string::npos);
    assert(output.find("primary_device_free_memory_mib=15360\n") != std::string::npos);
    assert(output.find("estimated_required_memory_mib=13312\n") != std::string::npos);
    assert(output.find("llm_quantization=Q8_0\n") != std::string::npos);
    assert(output.find("vision_model=/models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf\n") !=
           std::string::npos);
    assert(output.find("audio_model=/models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf\n") !=
           std::string::npos);
    assert(output.find("tts_model=/models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf\n") != std::string::npos);
    assert(output.find("token2wav_device=gpu:0\n") != std::string::npos);
    assert(output.find("token2wav_threads=8\n") != std::string::npos);
    assert(output.find("execution=async,duplex,vpm_batch_encode,ctx_size=8192\n") != std::string::npos);
    assert(output.find("fallback=F16 rejected:") != std::string::npos);
}

static void test_effective_config_reports_autotune_output_consistency() {
    omni::effective_runtime_config config;
    config.autotune_cache_status = "applied";
    config.autotune_cache_reason = "matching cache";
    config.autotune_selected_candidate = "precision_q8_0";
    config.output_consistent_with_baseline = false;

    const std::string output = omni::format_effective_runtime_config(config);
    assert(output.find("autotune_selected_candidate=precision_q8_0\n") != std::string::npos);
    assert(output.find("output_consistent_with_baseline=false\n") != std::string::npos);
}

static const omni::module_placement & find_placement(const omni::effective_runtime_config & config,
                                                       const std::string & module) {
    for (const auto & placement : config.placements) {
        if (placement.module == module) {
            return placement;
        }
    }
    assert(false && "requested module placement is missing");
    return config.placements.front();
}

static void test_auto_assigns_modules_across_accelerators() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    hardware.devices.push_back({ "CUDA1", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });

    const auto result = omni::resolve_runtime_profile("auto", hardware, complete_inventory());

    assert(result.ok);
    assert(result.config.resolved_profile == "full_omni_multi_accelerator_f16");
    assert(find_placement(result.config, "llm").device == "CUDA0");
    assert(find_placement(result.config, "vision").device == "CUDA1");
    assert(find_placement(result.config, "audio").device == "CUDA1");
    assert(find_placement(result.config, "tts").device == "CUDA0");
    assert(find_placement(result.config, "projector").device == "CUDA1");
    assert(find_placement(result.config, "token2wav").device == "CUDA0");
    assert(find_placement(result.config, "vision").status == "active");
}

static void test_unknown_module_device_fails_closed() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    const auto inventory = complete_inventory();
    omni::runtime_profile_overrides overrides;
    overrides.vision_device = "CUDA9";

    const auto result = omni::resolve_runtime_profile("auto", hardware, inventory, overrides);

    assert(!result.ok);
    assert(result.error.find("vision") != std::string::npos);
    assert(result.error.find("CUDA9") != std::string::npos);
}

static void test_model_inventory_only_reports_files_that_exist() {
    temporary_model_tree tree;
    tree.touch("MiniCPM-o-4_5-F16.gguf");
    tree.touch("MiniCPM-o-4_5-Q4_K_M.gguf");
    tree.touch("vision/MiniCPM-o-4_5-vision-F16.gguf");
    tree.touch("audio/MiniCPM-o-4_5-audio-F16.gguf");
    tree.touch("tts/MiniCPM-o-4_5-tts-F16.gguf");
    tree.touch("tts/MiniCPM-o-4_5-projector-F16.gguf");
    tree.touch("token2wav-gguf/encoder.gguf");
    tree.touch("token2wav-gguf/flow_matching.gguf");
    tree.touch("token2wav-gguf/flow_extra.gguf");
    tree.touch("token2wav-gguf/hifigan2.gguf");
    tree.touch("token2wav-gguf/prompt_cache.gguf");

    const auto inventory = omni::discover_model_inventory(tree.root.string());

    assert(inventory.root == tree.root.string());
    assert(inventory.llm_f16 == (tree.root / "MiniCPM-o-4_5-F16.gguf").string());
    assert(inventory.llm_q8_0.empty());
    assert(inventory.llm_q4_k_m == (tree.root / "MiniCPM-o-4_5-Q4_K_M.gguf").string());
    assert(inventory.vision == (tree.root / "vision/MiniCPM-o-4_5-vision-F16.gguf").string());
    assert(inventory.token2wav_prompt_cache == (tree.root / "token2wav-gguf/prompt_cache.gguf").string());
}

static void test_omni_runtime_profile_arguments_are_parsed() {
    common_params            params;
    std::vector<std::string> arguments = {
        "llama-omni-server",        "--profile", "auto", "--model-dir", "/models/MiniCPM-o-4_5-gguf",
        "--autotune", "refresh", "--autotune-cache", "/tmp/omni-autotune.json",
        "--autotune-rounds", "5", "--autotune-warmup-rounds", "2", "--autotune-allow-quantized",
        "--token2wav-threads", "32", "--print-effective-config",
    };
    std::vector<char *> argv;
    for (auto & argument : arguments) {
        argv.push_back(argument.data());
    }

    const bool parsed =
        common_params_parse(static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_OMNI_SERVER);

    assert(parsed);
    assert(params.omni_runtime_profile.profile == "auto");
    assert(params.omni_runtime_profile.model_dir == "/models/MiniCPM-o-4_5-gguf");
    assert(params.omni_runtime_profile.autotune == "refresh");
    assert(params.omni_runtime_profile.autotune_cache == "/tmp/omni-autotune.json");
    assert(params.omni_runtime_profile.autotune_rounds == 5);
    assert(params.omni_runtime_profile.autotune_warmup_rounds == 2);
    assert(params.omni_runtime_profile.autotune_allow_quantized);
    assert(params.omni_runtime_profile.token2wav_threads == 32);
    assert(params.omni_runtime_profile.print_effective_config);
}

static bool parse_omni_server_arguments(std::vector<std::string> arguments) {
    common_params        params;
    std::vector<char *> argv;
    for (auto & argument : arguments) {
        argv.push_back(argument.data());
    }
    return common_params_parse(static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_OMNI_SERVER);
}

static void test_omni_runtime_profile_argument_ranges_are_validated() {
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--autotune-rounds", "0" }));
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--autotune-rounds", "11" }));
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--autotune-warmup-rounds", "4" }));
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--token2wav-threads", "0" }));
}

int main() {
    test_auto_selects_f16_when_accelerator_memory_is_sufficient();
    test_auto_falls_back_to_q8_when_f16_does_not_fit();
    test_auto_falls_back_to_q4_when_q8_does_not_fit();
    test_auto_reports_the_missing_omni_module();
    test_auto_uses_cpu_compatible_profile_without_an_accelerator();
    test_zero_memory_accelerator_is_not_misclassified_as_cpu_only();
    test_explicit_values_override_auto_recommendations();
    test_explicit_model_works_without_recommended_llm_candidates();
    test_effective_config_explains_modules_and_fallbacks();
    test_effective_config_reports_autotune_output_consistency();
    test_auto_assigns_modules_across_accelerators();
    test_unknown_module_device_fails_closed();
    test_model_inventory_only_reports_files_that_exist();
    test_omni_runtime_profile_arguments_are_parsed();
    test_omni_runtime_profile_argument_ranges_are_validated();
    return 0;
}
