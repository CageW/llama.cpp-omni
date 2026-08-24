#include "runtime-autotune-runner.h"
#include "runtime-autotune.h"
#include "runtime-hash.h"

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static constexpr uint64_t GiB = 1024ULL * 1024ULL * 1024ULL;

class temporary_autotune_tree {
  public:
    temporary_autotune_tree() {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() / ("omni-runtime-autotune-test-" + std::to_string(unique));
        std::filesystem::create_directories(root);
    }

    ~temporary_autotune_tree() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    std::filesystem::path write(const std::string & name, const std::string & contents = "test") const {
        const auto path = root / name;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary);
        file << contents;
        return path;
    }

    std::filesystem::path root;
};

static omni::autotune_build_identity test_build() {
    return { 999, "test-commit", "test-target" };
}

static omni::hardware_snapshot test_hardware() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 80 * GiB, 96 * GiB, "GPU" });
    hardware.devices.push_back({ "CUDA1", "NVIDIA H20", true, 82 * GiB, 96 * GiB, "GPU" });
    return hardware;
}

static omni::autotune_fingerprint test_fingerprint(const temporary_autotune_tree & tree,
                                                   const omni::hardware_snapshot & hardware) {
    const auto  model   = tree.write("models/model.gguf");
    const auto  fixture = tree.write("fixtures/frame-00.bin");
    std::string error;
    const auto  fingerprint = omni::create_autotune_fingerprint(test_build(), hardware,
                                                                { model.string(), fixture.string() }, "auto", {}, error);
    assert(error.empty());
    return fingerprint;
}

static omni::effective_runtime_config test_runtime_config() {
    omni::effective_runtime_config config;
    config.requested_profile           = "auto";
    config.resolved_profile            = "autotune_encoder_split";
    config.reason                      = "measured calibration winner";
    config.primary_device_name         = "CUDA0";
    config.primary_device_description  = "NVIDIA H20";
    config.primary_device_free_memory  = 80 * GiB;
    config.primary_device_total_memory = 96 * GiB;
    config.estimated_required_memory   = 32 * GiB;
    config.llm_model                   = "/models/model.gguf";
    config.llm_quantization            = "F16";
    config.llm_device                  = "CUDA0";
    config.vision_model                = "/models/vision.gguf";
    config.vision_device               = "CUDA1";
    config.audio_model                 = "/models/audio.gguf";
    config.audio_device                = "CUDA1";
    config.tts_model                   = "/models/tts.gguf";
    config.tts_device                  = "CUDA0";
    config.projector_model             = "/models/projector.gguf";
    config.token2wav_model_dir         = "/models/token2wav";
    config.n_gpu_layers                = -2;
    config.tts_gpu_layers              = -1;
    config.token2wav_device            = "CUDA0";
    config.token2wav_threads           = 8;
    config.n_ctx                       = 8192;
    config.duplex_mode                 = true;
    config.async_mode                  = true;
    config.vpm_batch_encode            = true;
    config.placements.push_back({ "llm", config.llm_model, "F16", "CUDA0", "CUDA", "async", "active" });
    config.placements.push_back({ "vision", config.vision_model, "F16", "CUDA1", "CUDA", "async", "active" });
    config.placements.push_back({ "audio", config.audio_model, "F16", "CUDA1", "CUDA", "async", "active" });
    config.placements.push_back({ "tts", config.tts_model, "F16", "CUDA0", "CUDA", "async", "active" });
    config.placements.push_back({ "projector", config.projector_model, "F16", "CUDA1", "CUDA", "async", "active" });
    config.placements.push_back({ "token2wav", config.token2wav_model_dir, "F16", "CUDA0", "CUDA", "async", "active" });
    config.fallback_reasons.push_back("measured candidate improved score by 8.0%");
    return config;
}

static const omni::autotune_candidate_record & find_candidate(const omni::autotune_candidate_plan & plan,
                                                              const std::string &                   id) {
    for (const auto & candidate : plan.candidates) {
        if (candidate.id == id) {
            return candidate;
        }
    }
    assert(false && "autotune candidate is missing");
    return plan.candidates.front();
}

static const omni::module_placement & find_placement(const omni::effective_runtime_config & config,
                                                     const std::string &                    module) {
    for (const auto & placement : config.placements) {
        if (placement.module == module) {
            return placement;
        }
    }
    assert(false && "module placement is missing");
    return config.placements.front();
}

static omni::autotune_calibration_observation valid_observation() {
    omni::autotune_calibration_observation observation;
    observation.modules_initialized        = true;
    observation.samples_finite             = true;
    observation.metrics.frame_total_p95_ms = 800.0;
    observation.metrics.encoder_p95_ms     = 140.0;
    observation.metrics.decode_p95_ms      = 400.0;
    observation.metrics.ttft_p95_ms        = 260.0;
    observation.metrics.first_audio_p95_ms = 500.0;
    observation.metrics.tts_rtf            = 0.5;
    observation.metrics.frame_count        = 12;
    observation.metrics.speak_count        = 1;
    observation.metrics.final_audio_count  = 1;
    observation.metrics.audio_sample_count = 24000;
    observation.metrics.timeout_count      = 0;
    observation.response_sha256            = omni::normalized_response_sha256("response");
    observation.tts_tokens_sha256          = omni::tts_token_sequence_sha256({ 1, 2, -1 });
    observation.tts_token_count            = 42;
    return observation;
}

static void test_fingerprint_changes_when_device_total_memory_changes() {
    temporary_autotune_tree tree;
    auto                    hardware = test_hardware();
    const auto              before   = test_fingerprint(tree, hardware);

    hardware.devices.at(1).total_memory += GiB;
    const auto after = test_fingerprint(tree, hardware);

    assert(before != after);
}

static void test_fingerprint_changes_when_ggml_device_type_changes() {
    temporary_autotune_tree tree;
    auto                    hardware = test_hardware();
    hardware.devices.at(0).type      = "GPU";
    const auto before                = test_fingerprint(tree, hardware);

    hardware.devices.at(0).type = "ACCEL";
    const auto after            = test_fingerprint(tree, hardware);

    assert(before != after);
}

static void test_fingerprint_changes_when_build_identity_changes() {
    temporary_autotune_tree tree;
    const auto              model = tree.write("models/model.gguf");
    std::string             error;
    const auto              before =
        omni::create_autotune_fingerprint(test_build(), test_hardware(), { model.string() }, "auto", {}, error);
    auto changed_build   = test_build();
    changed_build.commit = "different-commit";
    const auto after =
        omni::create_autotune_fingerprint(changed_build, test_hardware(), { model.string() }, "auto", {}, error);

    assert(error.empty());
    assert(before != after);
}

static void test_fingerprint_changes_when_input_file_metadata_changes() {
    temporary_autotune_tree tree;
    const auto              model = tree.write("models/model.gguf", "before");
    std::string             error;
    const auto              before =
        omni::create_autotune_fingerprint(test_build(), test_hardware(), { model.string() }, "auto", {}, error);
    tree.write("models/model.gguf", "after-with-a-different-size");
    const auto after =
        omni::create_autotune_fingerprint(test_build(), test_hardware(), { model.string() }, "auto", {}, error);

    assert(error.empty());
    assert(before != after);
}

static void test_fingerprint_changes_when_an_explicit_override_changes() {
    temporary_autotune_tree         tree;
    const auto                      model = tree.write("models/model.gguf");
    omni::runtime_profile_overrides before_overrides;
    before_overrides.n_ctx = 4096;
    omni::runtime_profile_overrides after_overrides;
    after_overrides.n_ctx = 8192;
    std::string error;
    const auto  before = omni::create_autotune_fingerprint(test_build(), test_hardware(), { model.string() }, "auto",
                                                           before_overrides, error);
    const auto  after  = omni::create_autotune_fingerprint(test_build(), test_hardware(), { model.string() }, "auto",
                                                           after_overrides, error);

    assert(error.empty());
    assert(before != after);
}

static void test_fingerprint_changes_when_a_runtime_input_changes() {
    temporary_autotune_tree tree;
    const auto              model = tree.write("models/model.gguf");
    std::string             error;
    const auto              before =
        omni::create_autotune_fingerprint(test_build(), test_hardware(),
                                          {
                                              model.string()
    },
                                          "auto", {}, error, { { "n_batch", "512" }, { "threads", "8" } });
    const auto after =
        omni::create_autotune_fingerprint(test_build(), test_hardware(),
                                          {
                                              model.string()
    },
                                          "auto", {}, error, { { "n_batch", "1024" }, { "threads", "8" } });

    assert(error.empty());
    assert(before != after);
}

static void test_sha256_hashes_bytes_and_files() {
    temporary_autotune_tree tree;
    const auto              file     = tree.write("abc.txt", "abc");
    const std::string       expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert(omni::sha256_bytes("abc", 3) == expected);

    std::string digest;
    std::string error;
    assert(omni::sha256_file(file.string(), digest, error));
    assert(error.empty());
    assert(digest == expected);
    assert(omni::normalized_response_sha256("  hello\r\nworld\t") == omni::normalized_response_sha256("hello\nworld"));
    assert(omni::tts_token_sequence_sha256({ 1, 256, -1 }) != omni::tts_token_sequence_sha256({ 1, 256 }));
}

static void test_fingerprint_changes_when_runtime_identity_changes() {
    temporary_autotune_tree         tree;
    const auto                      model = tree.write("models/model.gguf");
    omni::autotune_runtime_identity before_identity;
    before_identity.executable_sha256    = "executable-a";
    before_identity.cuda_driver_version  = "12.8";
    before_identity.cuda_runtime_version = "12.8";
    before_identity.backend_libraries.push_back({ "/lib/libggml-cuda.so", 100, 123, "backend-a" });
    auto after_identity                             = before_identity;
    after_identity.backend_libraries.front().sha256 = "backend-b";

    std::string error;
    const auto before = omni::create_autotune_fingerprint(test_build(), test_hardware(), { model.string() }, "auto", {},
                                                          error, {}, before_identity);
    const auto after  = omni::create_autotune_fingerprint(test_build(), test_hardware(), { model.string() }, "auto", {},
                                                          error, {}, after_identity);

    assert(error.empty());
    assert(before != after);
}

static void test_phase_three_cache_schema_is_invalid() {
    temporary_autotune_tree     tree;
    const auto                  expected_fingerprint = test_fingerprint(tree, test_hardware());
    const auto                  cache_path           = tree.root / "runtime-autotune-v1.json";
    omni::autotune_cache_record record;
    record.schema_version             = 1;
    record.fingerprint                = expected_fingerprint;
    record.fingerprint.schema_version = 1;
    record.selected_candidate         = "encoder_split";
    record.candidates.push_back(
        omni::evaluate_autotune_candidate("encoder_split", test_runtime_config(), valid_observation()));

    std::string error;
    assert(omni::write_autotune_cache(cache_path.string(), record, error));
    const auto read = omni::read_autotune_cache(cache_path.string(), expected_fingerprint);

    assert(read.status == omni::autotune_cache_status::invalid);
    assert(read.reason.find("schema") != std::string::npos);
}

static void test_memory_samples_use_nonnegative_deltas_and_track_peak() {
    std::vector<omni::autotune_memory_sample> samples;
    omni::append_autotune_memory_sample(samples, "before_load",
                                        {
                                            { "CUDA0", 100 },
                                            { "CUDA1", 200 }
    });
    omni::append_autotune_memory_sample(samples, "after_llm",
                                        {
                                            { "CUDA0", 120 },
                                            { "CUDA1", 150 }
    });
    omni::append_autotune_memory_sample(samples, "end_measured_turn",
                                        {
                                            { "CUDA0", 70  },
                                            { "CUDA1", 160 }
    });

    assert(samples.at(1).used_delta_bytes.at("CUDA0") == 0);
    assert(samples.at(1).used_delta_bytes.at("CUDA1") == 50);
    assert(samples.at(1).incremental_used_bytes.at("CUDA0") == 0);
    assert(samples.at(1).incremental_used_bytes.at("CUDA1") == 50);
    assert(samples.at(2).incremental_used_bytes.at("CUDA0") == 50);
    assert(samples.at(2).incremental_used_bytes.at("CUDA1") == 0);
    const auto peaks = omni::autotune_memory_peaks(samples);
    assert(peaks.at("CUDA0") == 30);
    assert(peaks.at("CUDA1") == 50);
}

static void test_measured_module_memory_uses_incremental_checkpoint_deltas() {
    omni::autotune_calibration_observation observation = valid_observation();
    omni::append_autotune_memory_sample(observation.memory_samples, "before_load",
                                        {
                                            { "CUDA0", 1000 }
    });
    omni::append_autotune_memory_sample(observation.memory_samples, "after_llm",
                                        {
                                            { "CUDA0", 800 }
    });
    omni::append_autotune_memory_sample(observation.memory_samples, "after_tts",
                                        {
                                            { "CUDA0", 700 }
    });
    omni::autotune_candidate_record planned;
    planned.id           = "encoder_split";
    planned.config       = test_runtime_config();
    const auto candidate = omni::evaluate_autotune_candidate(planned, { observation });

    const auto measured = omni::autotune_measured_module_bytes({ candidate });
    assert(measured.at("llm") == 200);
    assert(measured.at("tts") == 100);
}

static void test_evaluation_preserves_output_hashes_and_memory_evidence() {
    auto       observation        = valid_observation();
    const auto response_hash      = omni::normalized_response_sha256("response evidence");
    const auto tts_hash           = omni::tts_token_sequence_sha256({ 4, 5, -1 });
    observation.response_sha256   = response_hash;
    observation.tts_tokens_sha256 = tts_hash;
    observation.tts_token_count   = 42;
    omni::append_autotune_memory_sample(observation.memory_samples, "before_load",
                                        {
                                            { "CUDA0", 100 }
    });
    omni::append_autotune_memory_sample(observation.memory_samples, "end_measured_turn",
                                        {
                                            { "CUDA0", 70 }
    });

    const auto candidate = omni::evaluate_autotune_candidate("encoder_split", test_runtime_config(), observation);

    assert(candidate.response_sha256 == response_hash);
    assert(candidate.tts_tokens_sha256 == tts_hash);
    assert(candidate.tts_token_count == 42);
    assert(candidate.memory_samples.size() == 2);
    assert(candidate.device_memory_peaks.at("CUDA0") == 30);
}

static void test_cache_round_trip_preserves_the_selected_module_placement() {
    temporary_autotune_tree tree;
    const auto              fingerprint = test_fingerprint(tree, test_hardware());
    const auto              cache_path  = tree.root / "runtime-autotune.json";

    omni::autotune_cache_record record;
    record.fingerprint              = fingerprint;
    record.selected_candidate       = "encoder_split";
    record.audio_semantics_verified = false;
    record.created_at               = "2026-08-21T12:00:00Z";

    omni::autotune_candidate_record candidate;
    candidate.id                         = "encoder_split";
    candidate.config                     = test_runtime_config();
    candidate.metrics.frame_count        = 12;
    candidate.metrics.frame_total_p95_ms = 740.0;
    candidate.metrics.decode_p95_ms      = 380.0;
    candidate.metrics.first_audio_p95_ms = 420.0;
    candidate.metrics.tts_rtf            = 0.62;
    candidate.valid                      = true;
    candidate.score                      = 0.651;
    record.candidates.push_back(candidate);

    std::string error;
    assert(omni::write_autotune_cache(cache_path.string(), record, error));
    assert(error.empty());

    const auto loaded = omni::read_autotune_cache(cache_path.string(), fingerprint);
    assert(loaded.status == omni::autotune_cache_status::applied);
    assert(loaded.record.selected_candidate == "encoder_split");
    assert(!loaded.record.audio_semantics_verified);
    assert(loaded.record.candidates.size() == 1);
    assert(loaded.record.candidates.front().config.placements.size() == 6);
    assert(loaded.record.candidates.front().config.placements.at(1).module == "vision");
    assert(loaded.record.candidates.front().config.placements.at(1).device == "CUDA1");
}

static void test_malformed_cache_is_invalid() {
    temporary_autotune_tree tree;
    const auto              fingerprint = test_fingerprint(tree, test_hardware());
    const auto              cache_path  = tree.write("runtime-autotune.json", "not json");

    const auto loaded = omni::read_autotune_cache(cache_path.string(), fingerprint);

    assert(loaded.status == omni::autotune_cache_status::invalid);
    assert(!loaded.reason.empty());
}

static void test_cache_with_an_unknown_schema_is_invalid() {
    temporary_autotune_tree tree;
    const auto              fingerprint = test_fingerprint(tree, test_hardware());
    const auto              cache_path  = tree.root / "runtime-autotune.json";

    omni::autotune_cache_record record;
    record.schema_version     = omni::runtime_autotune_schema_version + 1;
    record.fingerprint        = fingerprint;
    record.selected_candidate = "encoder_split";
    omni::autotune_candidate_record candidate;
    candidate.id     = record.selected_candidate;
    candidate.config = test_runtime_config();
    candidate.valid  = true;
    record.candidates.push_back(candidate);

    std::string error;
    assert(omni::write_autotune_cache(cache_path.string(), record, error));
    const auto loaded = omni::read_autotune_cache(cache_path.string(), fingerprint);

    assert(loaded.status == omni::autotune_cache_status::invalid);
    assert(loaded.reason.find("schema") != std::string::npos);
}

static void test_cache_write_atomically_replaces_existing_content() {
    temporary_autotune_tree tree;
    const auto              fingerprint = test_fingerprint(tree, test_hardware());
    const auto              cache_path  = tree.root / "runtime-autotune.json";

    omni::autotune_cache_record first;
    first.fingerprint        = fingerprint;
    first.selected_candidate = "encoder_split";
    omni::autotune_candidate_record first_candidate;
    first_candidate.id     = first.selected_candidate;
    first_candidate.config = test_runtime_config();
    first_candidate.valid  = true;
    first.candidates.push_back(first_candidate);

    auto second                                       = first;
    second.selected_candidate                         = "consolidated";
    second.candidates.front().id                      = second.selected_candidate;
    second.candidates.front().config.resolved_profile = "autotune_consolidated";

    std::string error;
    assert(omni::write_autotune_cache(cache_path.string(), first, error));
    assert(omni::write_autotune_cache(cache_path.string(), second, error));

    const auto loaded = omni::read_autotune_cache(cache_path.string(), fingerprint);
    assert(loaded.status == omni::autotune_cache_status::applied);
    assert(loaded.record.selected_candidate == "consolidated");
    for (const auto & entry : std::filesystem::directory_iterator(tree.root)) {
        assert(entry.path().filename().string().find("runtime-autotune.json.tmp-") != 0);
    }
}

static void test_cache_with_an_unavailable_selected_device_is_invalid() {
    temporary_autotune_tree tree;
    const auto              fingerprint = test_fingerprint(tree, test_hardware());
    const auto              cache_path  = tree.root / "runtime-autotune.json";

    omni::autotune_cache_record record;
    record.fingerprint        = fingerprint;
    record.selected_candidate = "encoder_split";
    omni::autotune_candidate_record candidate;
    candidate.id     = record.selected_candidate;
    candidate.config = test_runtime_config();
    for (auto & placement : candidate.config.placements) {
        if (placement.module == "vision") {
            placement.device = "CUDA9";
        }
    }
    candidate.valid = true;
    record.candidates.push_back(candidate);

    std::string error;
    assert(omni::write_autotune_cache(cache_path.string(), record, error));
    const auto loaded = omni::read_autotune_cache(cache_path.string(), fingerprint);

    assert(loaded.status == omni::autotune_cache_status::invalid);
    assert(loaded.reason.find("CUDA9") != std::string::npos);
}

static omni::autotune_cache_record test_cache_for_plan(const omni::autotune_candidate_plan & plan) {
    omni::autotune_cache_record record;
    record.fingerprint.runtime_inputs["autotune_rounds"] = "3";
    record.skipped_reasons                               = plan.skipped_reasons;
    for (const auto & candidate : plan.candidates) {
        auto evaluated = omni::evaluate_autotune_candidate(
            candidate, { valid_observation(), valid_observation(), valid_observation() });
        if (candidate.config.token2wav_device == "cpu") {
            evaluated.screening_performed = true;
            evaluated.screening           = omni::evaluate_autotune_cpu_screening(valid_observation());
            evaluated.screening_rejected  = false;
        }
        record.candidates.push_back(std::move(evaluated));
    }
    assert(omni::apply_autotune_output_consistency(record.candidates, plan.baseline_candidate));
    const auto selection = omni::select_autotune_candidate(record.candidates, plan.baseline_candidate);
    assert(selection.ok);
    record.selected_candidate = selection.selected_candidate;
    record.selection_reason   = selection.reason;
    return record;
}

static void test_cache_plan_validation_rejects_a_tampered_scalar_device() {
    const auto plan   = omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {});
    auto       record = test_cache_for_plan(plan);
    for (auto & candidate : record.candidates) {
        if (candidate.id == record.selected_candidate) {
            candidate.config.llm_device = "CUDA9";
        }
    }

    std::string error;
    assert(!omni::validate_autotune_cache_against_plan(record, plan, error));
    assert(error.find("configuration") != std::string::npos);
}

static void test_cache_plan_validation_rejects_a_tampered_score() {
    const auto plan   = omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {});
    auto       record = test_cache_for_plan(plan);
    record.candidates.front().score += 0.1;

    std::string error;
    assert(!omni::validate_autotune_cache_against_plan(record, plan, error));
    assert(error.find("score") != std::string::npos);
}

static void test_cache_plan_validation_rejects_a_tampered_raw_round_score() {
    const auto plan   = omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {});
    auto       record = test_cache_for_plan(plan);
    record.candidates.front().rounds.front().score += 0.1;

    std::string error;
    assert(!omni::validate_autotune_cache_against_plan(record, plan, error));
    assert(error.find("round") != std::string::npos);
}

static void test_cache_plan_validation_rejects_a_valid_candidate_marked_invalid() {
    const auto plan      = omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {});
    auto       record    = test_cache_for_plan(plan);
    auto &     candidate = record.candidates.front();
    assert(candidate.id != record.selected_candidate);
    candidate.valid            = false;
    candidate.rejection_reason = "fabricated calibration failure";

    std::string error;
    assert(!omni::validate_autotune_cache_against_plan(record, plan, error));
    assert(error.find("validity") != std::string::npos);
}

static void test_cache_plan_validation_accepts_a_recorded_output_mismatch_rejection() {
    const auto plan         = omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {});
    auto       record       = test_cache_for_plan(plan);
    auto       changed      = valid_observation();
    changed.response_sha256 = omni::normalized_response_sha256("different placement response");
    const auto & planned    = find_candidate(plan, "consolidated");
    auto         replacement =
        omni::evaluate_autotune_candidate(planned, { changed, valid_observation(), valid_observation() });
    for (auto & candidate : record.candidates) {
        if (candidate.id == replacement.id) {
            candidate = replacement;
        }
    }
    assert(omni::apply_autotune_output_consistency(record.candidates, plan.baseline_candidate));
    const auto selection = omni::select_autotune_candidate(record.candidates, plan.baseline_candidate);
    assert(selection.ok);
    record.selected_candidate = selection.selected_candidate;
    record.selection_reason   = selection.reason;

    std::string error;
    assert(omni::validate_autotune_cache_against_plan(record, plan, error));
    assert(error.empty());
}

static void test_cache_plan_validation_accepts_a_cpu_screening_rejection_without_measured_rounds() {
    const auto plan                  = omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {});
    auto       record                = test_cache_for_plan(plan);
    auto       screening_observation = valid_observation();
    screening_observation.metrics.tts_rtf = 1.5;
    for (auto & candidate : record.candidates) {
        if (candidate.config.token2wav_device != "cpu") {
            continue;
        }
        const auto & planned          = find_candidate(plan, candidate.id);
        candidate                     = planned;
        candidate.screening_performed = true;
        candidate.screening_rejected  = true;
        candidate.screening           = omni::evaluate_autotune_cpu_screening(screening_observation);
        candidate.valid               = false;
        candidate.rejection_reason    = "CPU screening rejected candidate: " + candidate.screening.rejection_reason;
        break;
    }
    assert(omni::apply_autotune_output_consistency(record.candidates, plan.baseline_candidate));
    const auto selection = omni::select_autotune_candidate(record.candidates, plan.baseline_candidate);
    assert(selection.ok);
    record.selected_candidate = selection.selected_candidate;
    record.selection_reason   = selection.reason;

    std::string error;
    assert(omni::validate_autotune_cache_against_plan(record, plan, error));
    assert(error.empty());
}

static void test_runtime_profile_controls_session_options_when_present() {
    auto config              = test_runtime_config();
    config.duplex_mode       = true;
    config.tts_gpu_layers    = -1;
    config.token2wav_device  = "gpu:0";
    config.token2wav_threads = 32;

    const auto profiled = omni::resolve_runtime_session_options(&config, /*requested_duplex_mode=*/false,
                                                                /*requested_tts_gpu_layers=*/100, "cpu",
                                                                /*requested_token2wav_threads=*/4);
    assert(profiled.duplex_mode);
    assert(profiled.tts_gpu_layers == -1);
    assert(profiled.token2wav_device == "gpu:0");
    assert(profiled.token2wav_threads == 32);

    const auto legacy = omni::resolve_runtime_session_options(nullptr, /*requested_duplex_mode=*/false,
                                                              /*requested_tts_gpu_layers=*/100, "cpu",
                                                              /*requested_token2wav_threads=*/4);
    assert(!legacy.duplex_mode);
    assert(legacy.tts_gpu_layers == 100);
    assert(legacy.token2wav_device == "cpu");
    assert(legacy.token2wav_threads == 4);
}

static omni::autotune_plan_options test_plan_options() {
    omni::autotune_plan_options options;
    options.inventory.llm_f16    = "/models/model-f16.gguf";
    options.inventory.llm_q8_0   = "/models/model-q8_0.gguf";
    options.inventory.llm_q4_k_m = "/models/model-q4_k_m.gguf";
    options.hardware_concurrency = 64;
    return options;
}

static void test_candidate_plan_contains_phase_four_placements() {
    const auto plan =
        omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {}, test_plan_options());

    assert(plan.candidates.size() == 11);
    assert(plan.baseline_candidate == "encoder_split");

    const auto & consolidated = find_candidate(plan, "consolidated");
    for (const auto & placement : consolidated.config.placements) {
        assert(placement.device == "CUDA0");
    }

    const auto & split = find_candidate(plan, "encoder_split");
    assert(find_placement(split.config, "llm").device == "CUDA0");
    assert(find_placement(split.config, "vision").device == "CUDA1");
    assert(find_placement(split.config, "audio").device == "CUDA1");
    assert(find_placement(split.config, "tts").device == "CUDA0");
    assert(find_placement(split.config, "projector").device == "CUDA1");
    assert(find_placement(split.config, "token2wav").device == "CUDA0");

    const auto & local_projector = find_candidate(plan, "encoder_split_local_projector");
    assert(find_placement(local_projector.config, "llm").device == "CUDA0");
    assert(find_placement(local_projector.config, "vision").device == "CUDA1");
    assert(find_placement(local_projector.config, "audio").device == "CUDA1");
    assert(find_placement(local_projector.config, "tts").device == "CUDA0");
    assert(find_placement(local_projector.config, "projector").device == "CUDA0");
    assert(find_placement(local_projector.config, "token2wav").device == "CUDA0");
    assert(local_projector.config.token2wav_device == "gpu:0");

    const auto & token2wav_split = find_candidate(plan, "token2wav_split");
    assert(find_placement(token2wav_split.config, "llm").device == "CUDA0");
    assert(find_placement(token2wav_split.config, "vision").device == "CUDA1");
    assert(find_placement(token2wav_split.config, "audio").device == "CUDA1");
    assert(find_placement(token2wav_split.config, "tts").device == "CUDA0");
    assert(find_placement(token2wav_split.config, "projector").device == "CUDA1");
    assert(find_placement(token2wav_split.config, "token2wav").device == "CUDA1");
    assert(token2wav_split.config.token2wav_device == "gpu:1");

    const auto & output_pipeline = find_candidate(plan, "output_pipeline_split");
    assert(find_placement(output_pipeline.config, "llm").device == "CUDA0");
    assert(find_placement(output_pipeline.config, "vision").device == "CUDA1");
    assert(find_placement(output_pipeline.config, "audio").device == "CUDA1");
    assert(find_placement(output_pipeline.config, "tts").device == "CUDA1");
    assert(find_placement(output_pipeline.config, "projector").device == "CUDA1");
    assert(find_placement(output_pipeline.config, "token2wav").device == "CUDA1");

    const auto & cpu_hybrid = find_candidate(plan, "gpu_cpu_hybrid_t32");
    assert(find_placement(cpu_hybrid.config, "llm").device == "CUDA0");
    assert(find_placement(cpu_hybrid.config, "vision").device == "CUDA1");
    assert(find_placement(cpu_hybrid.config, "audio").device == "CUDA1");
    assert(find_placement(cpu_hybrid.config, "tts").device == "CUDA0");
    assert(find_placement(cpu_hybrid.config, "projector").device == "CUDA0");
    assert(find_placement(cpu_hybrid.config, "token2wav").device == "cpu");
    assert(cpu_hybrid.config.token2wav_device == "cpu");
    assert(cpu_hybrid.config.token2wav_threads == 32);

    const auto & q8 = find_candidate(plan, "precision_q8_0");
    assert(q8.stage == "precision");
    assert(!q8.selectable);
    assert(q8.advisory);
    assert(q8.config.llm_model == "/models/model-q8_0.gguf");
    assert(q8.config.llm_quantization == "Q8_0");
}

static void test_quantized_candidate_can_be_selectable_when_enabled() {
    auto options            = test_plan_options();
    options.allow_quantized = true;
    const auto plan         = omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {}, options);

    assert(find_candidate(plan, "precision_q8_0").selectable);
    assert(!find_candidate(plan, "precision_q8_0").advisory);
}

static void test_explicit_model_disables_precision_candidates() {
    auto options           = test_plan_options();
    options.model_explicit = true;
    const auto plan        = omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {}, options);

    for (const auto & candidate : plan.candidates) {
        assert(candidate.stage != "precision");
    }
}

static void test_cpu_thread_candidates_are_capped_and_deduplicated() {
    auto options                 = test_plan_options();
    options.inventory            = {};
    options.hardware_concurrency = 12;
    const auto plan = omni::create_autotune_candidate_plan(test_runtime_config(), test_hardware(), {}, options);

    assert(find_candidate(plan, "gpu_cpu_hybrid_t8").config.token2wav_threads == 8);
    assert(find_candidate(plan, "gpu_cpu_hybrid_t16").config.token2wav_threads == 12);
    size_t cpu_candidates = 0;
    for (const auto & candidate : plan.candidates) {
        cpu_candidates += candidate.id.rfind("gpu_cpu_hybrid_", 0) == 0 ? 1 : 0;
    }
    assert(cpu_candidates == 2);
}

static void test_candidate_plan_skips_split_candidates_without_a_secondary_accelerator() {
    auto hardware = test_hardware();
    hardware.devices.pop_back();

    const auto plan = omni::create_autotune_candidate_plan(test_runtime_config(), hardware, {});

    assert(plan.candidates.size() == 1);
    assert(plan.candidates.front().id == "consolidated");
}

static void test_memory_headroom_filters_challenger_but_keeps_baseline() {
    temporary_autotune_tree tree;
    auto                    config = test_runtime_config();
    config.llm_model               = tree.write("model.gguf").string();
    config.vision_model            = tree.write("vision.gguf").string();
    config.audio_model             = tree.write("audio.gguf").string();
    config.tts_model               = tree.write("tts.gguf").string();
    config.projector_model         = tree.write("projector.gguf").string();
    config.token2wav_model_dir     = (tree.root / "token2wav").string();
    tree.write("token2wav/encoder.gguf");

    auto hardware                      = test_hardware();
    hardware.devices.at(1).free_memory = 2 * GiB;
    omni::autotune_plan_options options;
    options.hardware_concurrency = 64;
    const auto plan              = omni::create_autotune_candidate_plan(config, hardware, {}, options);

    assert(plan.baseline_candidate == "encoder_split");
    find_candidate(plan, "encoder_split");
    bool output_pipeline_present = false;
    for (const auto & candidate : plan.candidates) {
        output_pipeline_present = output_pipeline_present || candidate.id == "output_pipeline_split";
    }
    assert(!output_pipeline_present);
    bool explained = false;
    for (const auto & reason : plan.skipped_reasons) {
        explained = explained || reason.find("output_pipeline_split") != std::string::npos;
    }
    assert(explained);
}

static void test_previous_measured_memory_can_raise_but_not_lower_static_estimates() {
    temporary_autotune_tree tree;
    auto                    config = test_runtime_config();
    config.llm_model               = tree.write("model.gguf").string();
    config.vision_model            = tree.write("vision.gguf").string();
    config.audio_model             = tree.write("audio.gguf").string();
    config.tts_model               = tree.write("tts.gguf").string();
    config.projector_model         = tree.write("projector.gguf").string();
    config.token2wav_model_dir     = (tree.root / "token2wav").string();
    tree.write("token2wav/encoder.gguf");
    omni::autotune_plan_options options;
    options.previous_measured_module_bytes["llm"] = 50 * GiB;

    const auto   plan     = omni::create_autotune_candidate_plan(config, test_hardware(), {}, options);
    const auto & baseline = find_candidate(plan, plan.baseline_candidate);
    assert(baseline.memory_estimates.at("CUDA0") > 50 * GiB);
}

static void test_candidate_plan_preserves_an_explicit_module_device() {
    auto config          = test_runtime_config();
    config.vision_device = "CUDA0";
    for (auto & placement : config.placements) {
        if (placement.module == "vision") {
            placement.device = "CUDA0";
        }
    }
    omni::runtime_profile_overrides overrides;
    overrides.vision_device = "CUDA0";

    const auto plan = omni::create_autotune_candidate_plan(config, test_hardware(), overrides);

    for (const auto & candidate : plan.candidates) {
        assert(find_placement(candidate.config, "vision").device == "CUDA0");
    }
}

static void test_valid_candidate_receives_policy_two_score() {
    const auto candidate = omni::evaluate_autotune_candidate("candidate", test_runtime_config(), valid_observation());

    assert(candidate.valid);
    assert(candidate.rejection_reason.empty());
    assert(candidate.score > 0.508999 && candidate.score < 0.509001);
}

static void test_multi_round_score_uses_median_plus_mad() {
    auto first                        = valid_observation();
    auto second                       = valid_observation();
    auto third                        = valid_observation();
    first.metrics.encoder_p95_ms      = 100.0;
    second.metrics.encoder_p95_ms     = 200.0;
    third.metrics.encoder_p95_ms      = 300.0;
    first.metrics.ttft_p95_ms         = 100.0;
    second.metrics.ttft_p95_ms        = 200.0;
    third.metrics.ttft_p95_ms         = 300.0;
    first.metrics.first_audio_p95_ms  = 400.0;
    second.metrics.first_audio_p95_ms = 500.0;
    third.metrics.first_audio_p95_ms  = 600.0;
    first.metrics.tts_rtf             = 0.4;
    second.metrics.tts_rtf            = 0.5;
    third.metrics.tts_rtf             = 0.6;
    omni::autotune_candidate_record planned;
    planned.id     = "encoder_split";
    planned.config = test_runtime_config();

    const auto candidate = omni::evaluate_autotune_candidate(planned, { first, second, third });

    assert(candidate.valid);
    assert(candidate.rounds.size() == 3);
    assert(candidate.metrics.encoder_p95_ms == 300.0);
    assert(candidate.metrics.ttft_p95_ms == 300.0);
    assert(candidate.metrics.first_audio_p95_ms == 600.0);
    assert(candidate.metrics.tts_rtf > 0.599999 && candidate.metrics.tts_rtf < 0.600001);
    assert(candidate.score > 0.629999 && candidate.score < 0.630001);
}

static void test_output_consistency_rejects_same_precision_mismatch() {
    omni::autotune_candidate_record baseline;
    baseline.id             = "encoder_split";
    baseline.config         = test_runtime_config();
    baseline                = omni::evaluate_autotune_candidate(baseline, { valid_observation(), valid_observation() });
    auto changed            = valid_observation();
    changed.response_sha256 = omni::normalized_response_sha256("different response");
    omni::autotune_candidate_record challenger;
    challenger.id     = "consolidated";
    challenger.config = test_runtime_config();
    challenger        = omni::evaluate_autotune_candidate(challenger, { valid_observation(), changed });
    std::vector<omni::autotune_candidate_record> candidates = { baseline, challenger };

    assert(omni::apply_autotune_output_consistency(candidates, "encoder_split"));
    assert(candidates.at(0).valid);
    assert(!candidates.at(1).valid);
    assert(candidates.at(1).rejection_reason.find("output hash") != std::string::npos);
}

static void test_quantized_output_mismatch_is_advisory() {
    omni::autotune_candidate_record baseline;
    baseline.id               = "encoder_split";
    baseline.config           = test_runtime_config();
    baseline                  = omni::evaluate_autotune_candidate(baseline, { valid_observation() });
    auto changed              = valid_observation();
    changed.tts_tokens_sha256 = omni::tts_token_sequence_sha256({ 9, -1 });
    omni::autotune_candidate_record quantized;
    quantized.id                                            = "precision_q8_0";
    quantized.stage                                         = "precision";
    quantized.selectable                                    = false;
    quantized.advisory                                      = true;
    quantized.config                                        = test_runtime_config();
    quantized                                               = omni::evaluate_autotune_candidate(quantized, { changed });
    std::vector<omni::autotune_candidate_record> candidates = { baseline, quantized };

    assert(omni::apply_autotune_output_consistency(candidates, "encoder_split"));
    assert(candidates.at(1).valid);
    assert(candidates.at(1).output_consistency_checked);
    assert(!candidates.at(1).output_matches_baseline);
}

static void test_selector_requires_every_paired_round_to_be_no_worse() {
    auto baseline  = omni::evaluate_autotune_candidate("encoder_split", test_runtime_config(), valid_observation());
    baseline.score = 0.5;
    baseline.rounds.resize(3);
    for (auto & round : baseline.rounds) {
        round.valid = true;
        round.score = 0.5;
    }
    auto challenger            = baseline;
    challenger.id              = "consolidated";
    challenger.score           = 0.45;
    challenger.rounds[0].score = 0.4;
    challenger.rounds[1].score = 0.55;
    challenger.rounds[2].score = 0.4;

    const auto selection = omni::select_autotune_candidate({ challenger, baseline }, "encoder_split");

    assert(selection.ok);
    assert(selection.selected_candidate == "encoder_split");
    assert(selection.reason.find("paired round") != std::string::npos);
}

static void test_candidate_order_rotates_deterministically() {
    const std::vector<std::string> ids = { "a", "b", "c" };
    assert(omni::rotated_autotune_candidate_order(ids, 0) == ids);
    assert(omni::rotated_autotune_candidate_order(ids, 1) == std::vector<std::string>({ "b", "c", "a" }));
    assert(omni::rotated_autotune_candidate_order(ids, 2) == std::vector<std::string>({ "c", "a", "b" }));
}

static void test_schedule_runs_cpu_screening_rotated_rounds_and_stops_after_failure() {
    omni::autotune_candidate_plan plan;
    plan.baseline_candidate = "a";
    for (const std::string & id : { "a", "b", "cpu" }) {
        omni::autotune_candidate_record candidate;
        candidate.id                      = id;
        candidate.config                  = test_runtime_config();
        candidate.config.token2wav_device = id == "cpu" ? "cpu" : "CUDA0";
        plan.candidates.push_back(candidate);
    }

    std::vector<std::string> calls;
    const auto               result = omni::execute_autotune_schedule(
        plan, 3, 2,
        [&](const omni::autotune_candidate_record & candidate, const omni::autotune_schedule_request & request,
            std::string &) {
            calls.push_back(candidate.id + ":" + (request.screening ? "screen" : std::to_string(request.round_index)) +
                                          ":" + std::to_string(request.warmup_rounds));
            auto observation = valid_observation();
            if (candidate.id == "cpu" && request.screening) {
                observation.metrics.tts_rtf = 1.5;
            }
            if (candidate.id == "b" && request.round_index == 1) {
                observation.metrics.frame_count = 11;
            }
            return observation;
        });

    const std::vector<std::string> expected_calls = {
        "cpu:screen:0", "a:0:2", "b:0:2", "b:1:2", "a:1:2", "a:2:2",
    };
    assert(calls == expected_calls);
    assert(result.candidates.size() == 3);
    assert(result.candidates[0].rounds.size() == 3);
    assert(result.candidates[1].rounds.size() == 2);
    assert(result.candidates[2].screening_rejected);
    assert(result.candidates[2].rounds.empty());
    assert(result.first_failed_round.at("b") == 1);
    assert(result.first_failed_round.at("cpu") == -1);
    assert(result.output_consistency_reference_available);
}

static void test_cpu_screening_rejects_only_structural_failure_or_rtf_at_least_one_point_five() {
    auto observation            = valid_observation();
    observation.metrics.tts_rtf = 1.4;
    assert(omni::evaluate_autotune_cpu_screening(observation).valid);

    observation.metrics.tts_rtf = 1.5;
    const auto slow             = omni::evaluate_autotune_cpu_screening(observation);
    assert(!slow.valid);
    assert(slow.rejection_reason.find("1.5") != std::string::npos);

    observation                     = valid_observation();
    observation.metrics.frame_count = 11;
    const auto incomplete           = omni::evaluate_autotune_cpu_screening(observation);
    assert(!incomplete.valid);
    assert(incomplete.rejection_reason.find("12 fixture frames") != std::string::npos);
}

static void test_realtime_constraint_rejects_a_candidate() {
    auto observation                       = valid_observation();
    observation.metrics.frame_total_p95_ms = 1000.0;

    const auto candidate = omni::evaluate_autotune_candidate("candidate", test_runtime_config(), observation);

    assert(!candidate.valid);
    assert(candidate.rejection_reason.find("frame-total P95") != std::string::npos);
}

static void test_candidate_requires_valid_output_hash_evidence() {
    auto observation = valid_observation();
    observation.response_sha256.clear();

    const auto candidate = omni::evaluate_autotune_candidate("candidate", test_runtime_config(), observation);

    assert(!candidate.valid);
    assert(candidate.rejection_reason.find("SHA-256 evidence") != std::string::npos);
}

static void test_selector_keeps_phase_two_baseline_when_improvement_is_below_five_percent() {
    auto baseline    = omni::evaluate_autotune_candidate("encoder_split", test_runtime_config(), valid_observation());
    auto challenger  = baseline;
    challenger.id    = "consolidated";
    challenger.score = baseline.score * 0.951;

    const auto selection = omni::select_autotune_candidate({ challenger, baseline }, "encoder_split");

    assert(selection.ok);
    assert(selection.selected_candidate == "encoder_split");
    assert(selection.reason.find("5%") != std::string::npos);
}

static void test_selector_uses_a_candidate_that_improves_by_at_least_five_percent() {
    auto baseline    = omni::evaluate_autotune_candidate("encoder_split", test_runtime_config(), valid_observation());
    auto challenger  = baseline;
    challenger.id    = "consolidated";
    challenger.score = baseline.score * 0.94;

    const auto selection = omni::select_autotune_candidate({ baseline, challenger }, "encoder_split");

    assert(selection.ok);
    assert(selection.selected_candidate == "consolidated");
}

static void test_selector_fails_when_every_candidate_is_invalid() {
    auto observation                  = valid_observation();
    observation.metrics.timeout_count = 1;
    const auto first = omni::evaluate_autotune_candidate("consolidated", test_runtime_config(), observation);
    observation.metrics.timeout_count     = 0;
    observation.metrics.final_audio_count = 0;
    const auto second = omni::evaluate_autotune_candidate("encoder_split", test_runtime_config(), observation);

    const auto selection = omni::select_autotune_candidate({ first, second }, "encoder_split");

    assert(!selection.ok);
    assert(selection.reason.find("no valid") != std::string::npos);
}

int main() {
    assert(omni::runtime_autotune_schema_version == 2);
    assert(omni::runtime_autotune_scoring_policy == 2);
    test_fingerprint_changes_when_device_total_memory_changes();
    test_fingerprint_changes_when_ggml_device_type_changes();
    test_fingerprint_changes_when_build_identity_changes();
    test_fingerprint_changes_when_input_file_metadata_changes();
    test_fingerprint_changes_when_an_explicit_override_changes();
    test_fingerprint_changes_when_a_runtime_input_changes();
    test_sha256_hashes_bytes_and_files();
    test_fingerprint_changes_when_runtime_identity_changes();
    test_phase_three_cache_schema_is_invalid();
    test_memory_samples_use_nonnegative_deltas_and_track_peak();
    test_measured_module_memory_uses_incremental_checkpoint_deltas();
    test_evaluation_preserves_output_hashes_and_memory_evidence();
    test_cache_round_trip_preserves_the_selected_module_placement();
    test_malformed_cache_is_invalid();
    test_cache_with_an_unknown_schema_is_invalid();
    test_cache_write_atomically_replaces_existing_content();
    test_cache_with_an_unavailable_selected_device_is_invalid();
    test_cache_plan_validation_rejects_a_tampered_scalar_device();
    test_cache_plan_validation_rejects_a_tampered_score();
    test_cache_plan_validation_rejects_a_tampered_raw_round_score();
    test_cache_plan_validation_rejects_a_valid_candidate_marked_invalid();
    test_cache_plan_validation_accepts_a_recorded_output_mismatch_rejection();
    test_cache_plan_validation_accepts_a_cpu_screening_rejection_without_measured_rounds();
    test_runtime_profile_controls_session_options_when_present();
    test_candidate_plan_contains_phase_four_placements();
    test_quantized_candidate_can_be_selectable_when_enabled();
    test_explicit_model_disables_precision_candidates();
    test_cpu_thread_candidates_are_capped_and_deduplicated();
    test_candidate_plan_skips_split_candidates_without_a_secondary_accelerator();
    test_memory_headroom_filters_challenger_but_keeps_baseline();
    test_previous_measured_memory_can_raise_but_not_lower_static_estimates();
    test_candidate_plan_preserves_an_explicit_module_device();
    test_valid_candidate_receives_policy_two_score();
    test_multi_round_score_uses_median_plus_mad();
    test_output_consistency_rejects_same_precision_mismatch();
    test_quantized_output_mismatch_is_advisory();
    test_selector_requires_every_paired_round_to_be_no_worse();
    test_candidate_order_rotates_deterministically();
    test_schedule_runs_cpu_screening_rotated_rounds_and_stops_after_failure();
    test_cpu_screening_rejects_only_structural_failure_or_rtf_at_least_one_point_five();
    test_realtime_constraint_rejects_a_candidate();
    test_candidate_requires_valid_output_hash_evidence();
    test_selector_keeps_phase_two_baseline_when_improvement_is_below_five_percent();
    test_selector_uses_a_candidate_that_improves_by_at_least_five_percent();
    test_selector_fails_when_every_candidate_is_invalid();
    return 0;
}
