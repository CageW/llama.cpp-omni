#include "runtime-autotune-runner.h"

#include "common.h"
#include "omni.h"
#include "runtime-hash.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

namespace omni {
namespace {

using clock_type = std::chrono::steady_clock;

struct audio_event {
    clock_type::time_point completed_at;
    int                    sample_count = 0;
    int                    sample_rate  = 0;
    bool                   is_final     = false;
};

struct audio_collector {
    std::mutex               mutex;
    std::condition_variable  cv;
    std::vector<audio_event> events;
    bool                     samples_finite = true;
    size_t                   final_count    = 0;
};

struct speak_event {
    clock_type::time_point pushed_at;
};

double percentile_95(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(values.size())));
    return values.at(std::max<size_t>(1, rank) - 1);
}

std::string indexed_fixture_path(const std::string & prefix, int32_t index, const char * extension) {
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "%04d%s", index, extension);
    return prefix + suffix;
}

std::map<std::string, uint64_t> capture_device_free_memory(const effective_runtime_config & config) {
    std::map<std::string, uint64_t> output;
    for (const auto & placement : config.placements) {
        if (placement.device == "cpu" || output.count(placement.device) > 0) {
            continue;
        }
        const auto device = ggml_backend_dev_by_name(placement.device.c_str());
        if (device == nullptr) {
            continue;
        }
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        if (total > 0) {
            output[placement.device] = free;
        }
    }
    return output;
}

}  // namespace

runtime_session_options resolve_runtime_session_options(const effective_runtime_config * config,
                                                        bool                             requested_duplex_mode,
                                                        int32_t                          requested_tts_gpu_layers,
                                                        const std::string &              requested_token2wav_device,
                                                        int32_t                          requested_token2wav_threads) {
    if (config != nullptr) {
        return { config->duplex_mode, config->tts_gpu_layers, config->token2wav_device, config->token2wav_threads };
    }
    return { requested_duplex_mode, requested_tts_gpu_layers, requested_token2wav_device, requested_token2wav_threads };
}

bool apply_effective_runtime_config(common_params &                  params,
                                    const effective_runtime_config & config,
                                    std::string &                    error) {
    error.clear();
    params.model.path       = config.llm_model;
    params.n_gpu_layers     = config.n_gpu_layers;
    params.n_ctx            = config.n_ctx;
    params.vpm_model        = config.vision_model;
    params.apm_model        = config.audio_model;
    params.tts_model        = config.tts_model;
    params.projector_model  = config.projector_model;
    params.vpm_batch_encode = config.vpm_batch_encode;

    const auto separator = config.tts_model.find_last_of("/\\");
    params.tts_bin_dir   = separator == std::string::npos ? "." : config.tts_model.substr(0, separator);

    ggml_backend_dev_t llm_device = config.llm_device == "cpu" ?
                                        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) :
                                        ggml_backend_dev_by_name(config.llm_device.c_str());
    if (llm_device == nullptr) {
        error = "resolved LLM placement device is unavailable: " + config.llm_device;
        return false;
    }
    params.devices    = { llm_device, nullptr };
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu   = 0;
    if (config.llm_device == "cpu") {
        params.n_gpu_layers = 0;
    }
    return true;
}

autotune_calibration_observation run_autotune_candidate(const common_params &             base_params,
                                                        const autotune_candidate_record & candidate,
                                                        const autotune_fixture &          fixture,
                                                        const std::string &               run_directory,
                                                        std::string &                     error,
                                                        int32_t                           warmup_rounds) {
    autotune_calibration_observation observation;
    error.clear();

    if (warmup_rounds < 0) {
        error = "autotune warmup round count must not be negative";
        return observation;
    }
    if (fixture.frame_count != 12) {
        error = "autotune fixture must contain exactly 12 frames";
        return observation;
    }
    for (int32_t index = 0; index < fixture.frame_count; ++index) {
        const auto audio = indexed_fixture_path(fixture.frame_prefix, index, ".wav");
        const auto image = indexed_fixture_path(fixture.frame_prefix, index, ".jpg");
        if (!std::filesystem::is_regular_file(audio) || !std::filesystem::is_regular_file(image)) {
            error = "missing autotune fixture pair at index " + std::to_string(index);
            return observation;
        }
    }
    if (!std::filesystem::is_regular_file(fixture.reference_audio)) {
        error = "missing autotune reference audio: " + fixture.reference_audio;
        return observation;
    }

    std::error_code filesystem_error;
    std::filesystem::create_directories(run_directory, filesystem_error);
    if (filesystem_error) {
        error = "cannot create autotune run directory: " + filesystem_error.message();
        return observation;
    }

    common_params params = base_params;
    if (!apply_effective_runtime_config(params, candidate.config, error)) {
        return observation;
    }
    params.n_parallel    = 1;
    params.sampling.seed = 42;

    auto memory_observer = [&](const std::string & checkpoint) {
        append_autotune_memory_sample(observation.memory_samples, checkpoint,
                                      capture_device_free_memory(candidate.config));
    };

    omni_context * context = omni_init(&params, 2, true, params.tts_bin_dir, candidate.config.tts_gpu_layers,
                                       candidate.config.token2wav_device, true, nullptr, nullptr, run_directory,
                                       &candidate.config, true, candidate.config.token2wav_threads, memory_observer);
    if (context == nullptr) {
        error = "one or more requested modules failed to initialize for candidate " + candidate.id;
        return observation;
    }
    observation.modules_initialized = true;
    context->async                  = true;
    context->ref_audio_path         = fixture.reference_audio;

    const auto initialization_memory_samples = observation.memory_samples;
    for (int32_t turn_index = 0; turn_index <= warmup_rounds; ++turn_index) {
        const bool measured_turn        = turn_index == warmup_rounds;
        observation                     = {};
        observation.modules_initialized = true;
        if (measured_turn) {
            observation.memory_samples = initialization_memory_samples;
        }

        audio_collector      audio;
        std::mutex           token_mutex;
        std::vector<int32_t> tts_tokens;
        context->tts_token_output_cb = [&](const int32_t * tokens, size_t count, bool is_final) {
            std::lock_guard<std::mutex> lock(token_mutex);
            if (tokens != nullptr && count > 0) {
                tts_tokens.insert(tts_tokens.end(), tokens, tokens + count);
            }
            if (is_final) {
                tts_tokens.push_back(-1);
            }
        };
        context->audio_output_cb = [&](const float * samples, int sample_count, int sample_rate, bool is_final) {
            audio_event event;
            event.completed_at = clock_type::now();
            event.sample_count = sample_count;
            event.sample_rate  = sample_rate;
            event.is_final     = is_final;

            bool finite = sample_count >= 0 && (sample_count == 0 || samples != nullptr);
            for (int index = 0; finite && index < sample_count; ++index) {
                finite = std::isfinite(samples[index]);
            }
            {
                std::lock_guard<std::mutex> lock(audio.mutex);
                audio.samples_finite = audio.samples_finite && finite;
                audio.events.push_back(event);
                audio.final_count += is_final ? 1 : 0;
            }
            audio.cv.notify_all();
        };

        if (!omni_duplex_session_begin(context, fixture.reference_audio, run_directory)) {
            error = "failed to start duplex calibration session for candidate " + candidate.id;
            omni_free(context);
            return observation;
        }

        std::vector<double>      frame_total_ms;
        std::vector<double>      encoder_ms;
        std::vector<double>      decode_ms;
        std::vector<double>      ttft_ms;
        std::vector<speak_event> speak_events;
        std::string              response_text;
        bool                     previous_was_speak = false;
        const auto               schedule_start     = clock_type::now();

        for (int32_t index = 0; index < fixture.frame_count; ++index) {
            std::this_thread::sleep_until(
                schedule_start + std::chrono::milliseconds(static_cast<int64_t>(index) * fixture.stream_interval_ms));

            OmniDuplexFrame frame;
            frame.aud_fname      = indexed_fixture_path(fixture.frame_prefix, index, ".wav");
            frame.img_fname      = indexed_fixture_path(fixture.frame_prefix, index, ".jpg");
            frame.user_seq       = index + 1;
            const auto pushed_at = clock_type::now();
            if (omni_duplex_push_frame(context, frame) < 0) {
                error = "failed to submit autotune fixture frame " + std::to_string(index);
                observation.metrics.timeout_count++;
                break;
            }

            OmniDuplexFrameResult result;
            if (!omni_duplex_wait_next_frame(context, &result, fixture.frame_timeout_ms) || !result.ok) {
                error = "autotune fixture frame " + std::to_string(index) + " failed or timed out";
                observation.metrics.timeout_count++;
                context->break_event.store(true);
                break;
            }

            observation.metrics.frame_count++;
            observation.metrics.speak_count += result.is_speak ? 1 : 0;
            frame_total_ms.push_back(result.ms_total);
            encoder_ms.push_back(result.ms_encoder);
            decode_ms.push_back(result.ms_decode);
            ttft_ms.push_back(result.ms_ttft);
            response_text += result.text;
            if (result.is_speak && !previous_was_speak) {
                speak_events.push_back({ pushed_at });
            }
            previous_was_speak = result.is_speak;
        }

        const bool session_drained = omni_duplex_session_end(context, fixture.frame_timeout_ms);
        if (!session_drained) {
            observation.metrics.timeout_count++;
            if (error.empty()) {
                error = "autotune duplex session did not drain before the timeout";
            }
        }

        bool needs_explicit_flush = false;
        if (session_drained && previous_was_speak) {
            std::lock_guard<std::mutex> lock(audio.mutex);
            needs_explicit_flush = audio.final_count < speak_events.size();
        }
        if (needs_explicit_flush && !omni_duplex_flush_tts(context)) {
            observation.metrics.timeout_count++;
            if (error.empty()) {
                error = "autotune could not close the final SPEAK turn";
            }
        }

        bool audio_completed = true;
        if (session_drained && error.empty() && !speak_events.empty()) {
            std::unique_lock<std::mutex> lock(audio.mutex);
            audio_completed = audio.cv.wait_for(lock, std::chrono::milliseconds(fixture.audio_timeout_ms),
                                                [&] { return audio.final_count >= speak_events.size(); });
        }
        if (!audio_completed) {
            observation.metrics.timeout_count++;
            context->break_event.store(true);
            if (error.empty()) {
                error = "autotune audio pipeline did not emit final audio before the timeout";
            }
        }

        if (measured_turn) {
            memory_observer("end_measured_turn");
        }
        std::vector<audio_event> audio_events;
        {
            std::lock_guard<std::mutex> lock(audio.mutex);
            audio_events               = audio.events;
            observation.samples_finite = audio.samples_finite;
        }
        {
            std::lock_guard<std::mutex> lock(token_mutex);
            observation.tts_token_count   = static_cast<int64_t>(tts_tokens.size());
            observation.tts_tokens_sha256 = tts_token_sequence_sha256(tts_tokens);
        }
        observation.response_sha256 = normalized_response_sha256(response_text);

        observation.metrics.frame_total_p95_ms = percentile_95(frame_total_ms);
        observation.metrics.encoder_p95_ms     = percentile_95(encoder_ms);
        observation.metrics.decode_p95_ms      = percentile_95(decode_ms);
        observation.metrics.ttft_p95_ms        = percentile_95(ttft_ms);

        std::vector<double> first_audio_ms;
        std::vector<double> turn_rtfs;
        size_t              audio_begin = 0;
        size_t              speak_index = 0;
        for (size_t index = 0; index < audio_events.size(); ++index) {
            const auto & event = audio_events[index];
            observation.metrics.audio_sample_count += event.sample_count;
            if (!event.is_final) {
                continue;
            }
            observation.metrics.final_audio_count++;
            if (speak_index < speak_events.size()) {
                const auto & speak = speak_events[speak_index++];
                first_audio_ms.push_back(
                    std::chrono::duration<double, std::milli>(audio_events[audio_begin].completed_at - speak.pushed_at)
                        .count());
                double duration_seconds = 0.0;
                for (size_t audio_index = audio_begin; audio_index <= index; ++audio_index) {
                    if (audio_events[audio_index].sample_rate > 0) {
                        duration_seconds += static_cast<double>(audio_events[audio_index].sample_count) /
                                            audio_events[audio_index].sample_rate;
                    }
                }
                const double wall_seconds = std::chrono::duration<double>(event.completed_at - speak.pushed_at).count();
                if (duration_seconds > 0.0) {
                    turn_rtfs.push_back(std::max(0.0, wall_seconds) / duration_seconds);
                }
            }
            audio_begin = index + 1;
        }
        observation.metrics.first_audio_p95_ms = percentile_95(first_audio_ms);
        if (!turn_rtfs.empty()) {
            double sum = 0.0;
            for (const double value : turn_rtfs) {
                sum += value;
            }
            observation.metrics.tts_rtf = sum / static_cast<double>(turn_rtfs.size());
        }
        if (!measured_turn) {
            const bool structurally_complete =
                error.empty() && observation.modules_initialized && observation.metrics.timeout_count == 0 &&
                observation.metrics.frame_count == fixture.frame_count && observation.metrics.speak_count > 0 &&
                observation.samples_finite && observation.metrics.audio_sample_count > 0 &&
                observation.metrics.final_audio_count > 0;
            if (!structurally_complete) {
                if (error.empty()) {
                    error = "autotune warmup turn failed structural validation for candidate " + candidate.id;
                }
                omni_free(context);
                return observation;
            }
            context->tts_token_output_cb = {};
            context->audio_output_cb     = {};
            if (!omni_reset_calibration_state(context)) {
                error = "autotune could not reset state after a warmup turn for candidate " + candidate.id;
                omni_free(context);
                return observation;
            }
        } else {
            context->tts_token_output_cb = {};
            context->audio_output_cb     = {};
            omni_free(context);
            return observation;
        }
    }
    omni_free(context);
    return observation;
}

}  // namespace omni
