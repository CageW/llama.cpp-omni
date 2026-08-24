#pragma once

#include "runtime-profile.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace omni {

static constexpr int32_t runtime_autotune_schema_version = 2;
static constexpr int32_t runtime_autotune_scoring_policy = 2;

struct autotune_build_identity {
    int32_t     build_number = 0;
    std::string commit;
    std::string target;
};

struct autotune_file_fingerprint {
    std::string path;
    uint64_t    size          = 0;
    int64_t     modified_time = 0;
};

struct autotune_backend_identity {
    std::string path;
    uint64_t    size          = 0;
    int64_t     modified_time = 0;
    std::string sha256;
};

struct autotune_runtime_identity {
    std::string                            executable_sha256;
    std::vector<autotune_backend_identity> backend_libraries;
    std::string                            cuda_driver_version  = "unavailable";
    std::string                            cuda_runtime_version = "unavailable";
};

struct autotune_device_fingerprint {
    std::string name;
    std::string description;
    bool        is_accelerator = false;
    uint64_t    total_memory   = 0;
    std::string type;
};

struct autotune_fingerprint {
    int32_t                                  schema_version = runtime_autotune_schema_version;
    int32_t                                  scoring_policy = runtime_autotune_scoring_policy;
    autotune_build_identity                  build;
    std::string                              requested_profile;
    runtime_profile_overrides                overrides;
    std::map<std::string, std::string>       runtime_inputs;
    autotune_runtime_identity                runtime_identity;
    std::vector<autotune_device_fingerprint> devices;
    std::vector<autotune_file_fingerprint>   files;
};

struct autotune_metrics {
    double  frame_total_p95_ms = 0.0;
    double  encoder_p95_ms     = 0.0;
    double  decode_p95_ms      = 0.0;
    double  ttft_p95_ms        = 0.0;
    double  first_audio_p95_ms = 0.0;
    double  tts_rtf            = 0.0;
    int32_t frame_count        = 0;
    int32_t speak_count        = 0;
    int32_t final_audio_count  = 0;
    int64_t audio_sample_count = 0;
    int32_t timeout_count      = 0;
};

struct autotune_memory_sample {
    std::string                     checkpoint;
    std::map<std::string, uint64_t> free_memory_bytes;
    std::map<std::string, uint64_t> used_delta_bytes;
    std::map<std::string, uint64_t> incremental_used_bytes;
};

struct autotune_round_record {
    int32_t                             round_index = 0;
    autotune_metrics                    metrics;
    std::vector<autotune_memory_sample> memory_samples;
    std::map<std::string, uint64_t>     device_memory_peaks;
    std::string                         response_sha256;
    std::string                         tts_tokens_sha256;
    int64_t                             tts_token_count     = 0;
    bool                                modules_initialized = false;
    bool                                samples_finite      = false;
    bool                                valid               = false;
    std::string                         execution_error;
    std::string                         rejection_reason;
    double                              score = 0.0;
};

struct autotune_candidate_record {
    std::string                         id;
    std::string                         stage = "placement";
    effective_runtime_config            config;
    autotune_metrics                    metrics;
    std::map<std::string, uint64_t>     memory_estimates;
    std::vector<autotune_round_record>  rounds;
    std::vector<autotune_memory_sample> memory_samples;
    std::map<std::string, uint64_t>     device_memory_peaks;
    std::string                         response_sha256;
    std::string                         tts_tokens_sha256;
    int64_t                             tts_token_count     = 0;
    bool                                selectable          = true;
    bool                                advisory            = false;
    bool                                screening_performed = false;
    bool                                screening_rejected  = false;
    autotune_round_record               screening;
    bool                                output_consistency_checked = false;
    bool                                output_matches_baseline    = false;
    bool                                modules_initialized        = false;
    bool                                samples_finite             = false;
    bool                                valid                      = false;
    std::string                         rejection_reason;
    double                              score = 0.0;
};

struct autotune_calibration_observation {
    autotune_metrics                    metrics;
    std::vector<autotune_memory_sample> memory_samples;
    std::string                         response_sha256;
    std::string                         tts_tokens_sha256;
    int64_t                             tts_token_count     = 0;
    bool                                modules_initialized = false;
    bool                                samples_finite      = false;
    std::string                         execution_error;
};

struct autotune_candidate_plan {
    std::string                            baseline_candidate;
    std::vector<autotune_candidate_record> candidates;
    std::vector<std::string>               skipped_reasons;
};

struct autotune_plan_options {
    model_inventory                 inventory;
    bool                            model_explicit       = false;
    bool                            allow_quantized      = false;
    uint32_t                        hardware_concurrency = 0;
    std::map<std::string, uint64_t> previous_measured_module_bytes;
};

struct autotune_selection_result {
    bool        ok = false;
    std::string selected_candidate;
    std::string reason;
};

struct autotune_schedule_request {
    bool    screening     = false;
    int32_t round_index   = 0;
    int32_t warmup_rounds = 0;
};

using autotune_run_callback = std::function<autotune_calibration_observation(const autotune_candidate_record &,
                                                                             const autotune_schedule_request &,
                                                                             std::string &)>;

struct autotune_schedule_result {
    std::vector<autotune_candidate_record> candidates;
    std::map<std::string, int32_t>         first_failed_round;
    bool                                   output_consistency_reference_available = false;
};

struct autotune_cache_record {
    int32_t                                schema_version = runtime_autotune_schema_version;
    autotune_fingerprint                   fingerprint;
    std::string                            selected_candidate;
    std::string                            selection_reason;
    std::vector<autotune_candidate_record> candidates;
    std::vector<std::string>               skipped_reasons;
    std::map<std::string, uint64_t>        memory_estimate_inputs;
    bool                                   audio_semantics_verified = false;
    std::string                            created_at;
};

enum class autotune_cache_status {
    applied,
    missing,
    stale,
    invalid,
};

struct autotune_cache_read_result {
    autotune_cache_status status = autotune_cache_status::missing;
    autotune_cache_record record;
    std::string           reason;
};

bool operator==(const autotune_build_identity & left, const autotune_build_identity & right);
bool operator==(const autotune_file_fingerprint & left, const autotune_file_fingerprint & right);
bool operator==(const autotune_backend_identity & left, const autotune_backend_identity & right);
bool operator==(const autotune_runtime_identity & left, const autotune_runtime_identity & right);
bool operator==(const autotune_device_fingerprint & left, const autotune_device_fingerprint & right);
bool operator==(const autotune_fingerprint & left, const autotune_fingerprint & right);
bool operator!=(const autotune_fingerprint & left, const autotune_fingerprint & right);

autotune_fingerprint create_autotune_fingerprint(const autotune_build_identity &            build,
                                                 const hardware_snapshot &                  hardware,
                                                 const std::vector<std::string> &           files,
                                                 const std::string &                        requested_profile,
                                                 const runtime_profile_overrides &          overrides,
                                                 std::string &                              error,
                                                 const std::map<std::string, std::string> & runtime_inputs   = {},
                                                 const autotune_runtime_identity &          runtime_identity = {});

autotune_runtime_identity detect_autotune_runtime_identity(std::string & error);

void                            append_autotune_memory_sample(std::vector<autotune_memory_sample> &   samples,
                                                              const std::string &                     checkpoint,
                                                              const std::map<std::string, uint64_t> & free_memory_bytes);
std::map<std::string, uint64_t> autotune_memory_peaks(const std::vector<autotune_memory_sample> & samples);
std::map<std::string, uint64_t> autotune_measured_module_bytes(
    const std::vector<autotune_candidate_record> & candidates);

bool write_autotune_cache(const std::string & path, const autotune_cache_record & record, std::string & error);

autotune_cache_read_result read_autotune_cache(const std::string &          path,
                                               const autotune_fingerprint & expected_fingerprint);

autotune_candidate_plan create_autotune_candidate_plan(const effective_runtime_config &  base_config,
                                                       const hardware_snapshot &         hardware,
                                                       const runtime_profile_overrides & overrides,
                                                       const autotune_plan_options &     options = {});

autotune_candidate_record evaluate_autotune_candidate(const std::string &                      id,
                                                      const effective_runtime_config &         config,
                                                      const autotune_calibration_observation & observation);
autotune_candidate_record evaluate_autotune_candidate(
    const autotune_candidate_record &                     planned,
    const std::vector<autotune_calibration_observation> & observations);

autotune_round_record evaluate_autotune_cpu_screening(const autotune_calibration_observation & observation);

bool apply_autotune_output_consistency(std::vector<autotune_candidate_record> & candidates,
                                       const std::string &                      baseline_candidate);

std::vector<std::string> rotated_autotune_candidate_order(const std::vector<std::string> & candidate_ids,
                                                          int32_t                          round_index);

autotune_schedule_result execute_autotune_schedule(const autotune_candidate_plan & plan,
                                                   int32_t                         measured_rounds,
                                                   int32_t                         warmup_rounds,
                                                   const autotune_run_callback &   run);

autotune_selection_result select_autotune_candidate(const std::vector<autotune_candidate_record> & candidates,
                                                    const std::string &                            baseline_candidate);

bool validate_autotune_cache_against_plan(const autotune_cache_record &   record,
                                          const autotune_candidate_plan & plan,
                                          std::string &                   error);

}  // namespace omni
