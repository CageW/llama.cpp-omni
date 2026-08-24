#pragma once

#include "runtime-autotune.h"

#include <string>

struct common_params;

namespace omni {

struct autotune_fixture {
    std::string frame_prefix;
    std::string reference_audio;
    int32_t     frame_count        = 12;
    int32_t     stream_interval_ms = 1000;
    int32_t     frame_timeout_ms   = 30000;
    int32_t     audio_timeout_ms   = 120000;
};

struct runtime_session_options {
    bool        duplex_mode       = false;
    int32_t     tts_gpu_layers    = 100;
    std::string token2wav_device  = "gpu:0";
    int32_t     token2wav_threads = 8;
};

runtime_session_options resolve_runtime_session_options(const effective_runtime_config * config,
                                                        bool                             requested_duplex_mode,
                                                        int32_t                          requested_tts_gpu_layers,
                                                        const std::string &              requested_token2wav_device,
                                                        int32_t requested_token2wav_threads = 8);

bool apply_effective_runtime_config(common_params &                  params,
                                    const effective_runtime_config & config,
                                    std::string &                    error);

autotune_calibration_observation run_autotune_candidate(const common_params &             base_params,
                                                        const autotune_candidate_record & candidate,
                                                        const autotune_fixture &          fixture,
                                                        const std::string &               run_directory,
                                                        std::string &                     error,
                                                        int32_t                           warmup_rounds = 0);

}  // namespace omni
