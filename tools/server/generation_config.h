#pragma once

#include "omni.h"

#include <nlohmann/json.hpp>

inline OmniGenerationConfig omni_generation_config_from_json(
        const nlohmann::json & values,
        OmniGenerationConfig config = {}) {
    if (!values.is_object()) {
        return config;
    }

    auto set_float = [&](const nlohmann::json & object,
                         const char * key,
                         float & target) {
        if (object.contains(key) && object.at(key).is_number()) {
            target = object.at(key).get<float>();
        }
    };
    auto set_int = [&](const nlohmann::json & object,
                       const char * key,
                       int & target) {
        if (object.contains(key) && object.at(key).is_number_integer()) {
            target = object.at(key).get<int>();
        }
    };

    set_float(values, "temperature", config.llm_sampling.temperature);
    set_float(values, "top_p", config.llm_sampling.top_p);
    set_int(values, "top_k", config.llm_sampling.top_k);
    set_float(values, "min_p", config.llm_sampling.min_p);
    if (values.contains("text_repetition_penalty")) {
        set_float(values, "text_repetition_penalty", config.llm_sampling.repetition_penalty);
    } else {
        set_float(values, "repetition_penalty", config.llm_sampling.repetition_penalty);
    }
    set_int(values, "text_repetition_window_size", config.llm_sampling.repetition_window);
    set_int(values, "min_tokens_to_keep", config.llm_sampling.min_tokens_to_keep);

    set_float(values, "tts_temperature", config.tts_sampling.temperature);
    set_float(values, "tts_top_p", config.tts_sampling.top_p);
    set_int(values, "tts_top_k", config.tts_sampling.top_k);
    set_float(values, "tts_min_p", config.tts_sampling.min_p);
    set_float(values, "tts_repetition_penalty", config.tts_sampling.repetition_penalty);
    set_int(values, "tts_repetition_window", config.tts_sampling.repetition_window);
    set_int(values, "tts_min_tokens_to_keep", config.tts_sampling.min_tokens_to_keep);

    set_int(values, "tts_n_timesteps", config.tts_n_timesteps);
    set_int(values, "n_timesteps", config.tts_n_timesteps);
    set_float(values, "token2wav_temperature", config.token2wav_temperature);
    set_int(values, "audio_token_chunk_size", config.audio_token_chunk_size);
    set_int(values, "audio_prelook_size", config.audio_prelook_size);
    set_int(values, "tts_max_audio_tokens", config.tts_max_audio_tokens);
    set_int(values, "tts_max_audio_tokens_per_text_token", config.tts_max_audio_tokens_per_text_token);
    set_int(values, "tts_min_audio_tokens", config.tts_min_audio_tokens);
    set_int(values, "duplex_tts_max_audio_tokens", config.duplex_tts_max_audio_tokens);
    set_int(values, "duplex_tts_min_audio_tokens", config.duplex_tts_min_audio_tokens);
    set_int(values, "llm_text_chunk_size", config.llm_text_chunk_size);
    set_int(values, "text_chunk_size", config.llm_text_chunk_size);
    set_int(values, "max_new_speak_tokens_per_chunk", config.max_new_speak_tokens_per_chunk);
    set_float(values, "listen_prob_scale", config.listen_prob_scale);
    set_int(values, "force_listen_count", config.force_listen_count);
    set_float(values, "length_penalty", config.length_penalty);

    if (values.contains("llm") && values.at("llm").is_object()) {
        const auto & llm = values.at("llm");
        set_float(llm, "temperature", config.llm_sampling.temperature);
        set_float(llm, "top_p", config.llm_sampling.top_p);
        set_int(llm, "top_k", config.llm_sampling.top_k);
        set_float(llm, "min_p", config.llm_sampling.min_p);
        set_float(llm, "repetition_penalty", config.llm_sampling.repetition_penalty);
        set_int(llm, "repetition_window", config.llm_sampling.repetition_window);
        set_int(llm, "min_tokens_to_keep", config.llm_sampling.min_tokens_to_keep);
        set_int(llm, "text_chunk_size", config.llm_text_chunk_size);
    }
    if (values.contains("tts") && values.at("tts").is_object()) {
        const auto & tts = values.at("tts");
        set_float(tts, "temperature", config.tts_sampling.temperature);
        set_float(tts, "top_p", config.tts_sampling.top_p);
        set_int(tts, "top_k", config.tts_sampling.top_k);
        set_float(tts, "min_p", config.tts_sampling.min_p);
        set_float(tts, "repetition_penalty", config.tts_sampling.repetition_penalty);
        set_int(tts, "repetition_window", config.tts_sampling.repetition_window);
        set_int(tts, "min_tokens_to_keep", config.tts_sampling.min_tokens_to_keep);
        set_int(tts, "n_timesteps", config.tts_n_timesteps);
        set_float(tts, "token2wav_temperature", config.token2wav_temperature);
        set_int(tts, "audio_token_chunk_size", config.audio_token_chunk_size);
        set_int(tts, "audio_prelook_size", config.audio_prelook_size);
        set_int(tts, "max_audio_tokens", config.tts_max_audio_tokens);
        set_int(tts, "max_audio_tokens_per_text_token", config.tts_max_audio_tokens_per_text_token);
        set_int(tts, "min_audio_tokens", config.tts_min_audio_tokens);
        set_int(tts, "duplex_max_audio_tokens", config.duplex_tts_max_audio_tokens);
        set_int(tts, "duplex_min_audio_tokens", config.duplex_tts_min_audio_tokens);
    }

    return config;
}
