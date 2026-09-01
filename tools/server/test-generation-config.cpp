#include "generation_config.h"

#include <cassert>
#include <cmath>

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1e-6f;
}

int main() {
    const OmniGenerationConfig defaults =
        omni_generation_config_from_json(nlohmann::json::object());

    assert(nearly_equal(defaults.llm_sampling.temperature, 0.7f));
    assert(defaults.llm_sampling.top_k == 100);
    assert(nearly_equal(defaults.llm_sampling.top_p, 0.8f));
    assert(nearly_equal(defaults.llm_sampling.repetition_penalty, 1.05f));
    assert(defaults.llm_sampling.repetition_window == 512);

    assert(nearly_equal(defaults.tts_sampling.temperature, 0.8f));
    assert(defaults.tts_sampling.top_k == 25);
    assert(nearly_equal(defaults.tts_sampling.top_p, 0.85f));
    assert(nearly_equal(defaults.tts_sampling.min_p, 0.01f));
    assert(nearly_equal(defaults.tts_sampling.repetition_penalty, 1.05f));
    assert(defaults.tts_sampling.repetition_window == 16);
    assert(defaults.tts_sampling.min_tokens_to_keep == 3);

    assert(defaults.tts_n_timesteps == 10);
    assert(defaults.audio_token_chunk_size == 25);
    assert(defaults.audio_prelook_size == 3);
    assert(defaults.llm_text_chunk_size == 10);
    assert(defaults.tts_max_audio_tokens == 500);
    assert(defaults.duplex_tts_max_audio_tokens == 26);
    assert(defaults.duplex_tts_min_audio_tokens == 26);
    assert(defaults.max_new_speak_tokens_per_chunk == 20);
    assert(defaults.force_listen_count == 0);

    const OmniGenerationConfig overridden =
        omni_generation_config_from_json({
            {"temperature", 0.6},
            {"top_k", 12},
            {"top_p", 0.75},
            {"text_repetition_penalty", 1.1},
            {"tts_temperature", 0.65},
            {"tts_top_k", 18},
            {"tts_top_p", 0.9},
            {"tts_repetition_penalty", 1.02},
            {"tts_min_p", 0.02},
            {"tts_n_timesteps", 6},
            {"audio_token_chunk_size", 24},
            {"audio_prelook_size", 4},
            {"llm_text_chunk_size", 8},
            {"tts_max_audio_tokens", 480},
            {"duplex_tts_max_audio_tokens", 28},
            {"duplex_tts_min_audio_tokens", 24},
            {"max_new_speak_tokens_per_chunk", 18},
            {"force_listen_count", 2},
            {"listen_prob_scale", 1.2},
        });

    assert(nearly_equal(overridden.llm_sampling.temperature, 0.6f));
    assert(overridden.llm_sampling.top_k == 12);
    assert(nearly_equal(overridden.llm_sampling.top_p, 0.75f));
    assert(nearly_equal(overridden.llm_sampling.repetition_penalty, 1.1f));
    assert(nearly_equal(overridden.tts_sampling.temperature, 0.65f));
    assert(overridden.tts_sampling.top_k == 18);
    assert(nearly_equal(overridden.tts_sampling.top_p, 0.9f));
    assert(nearly_equal(overridden.tts_sampling.repetition_penalty, 1.02f));
    assert(nearly_equal(overridden.tts_sampling.min_p, 0.02f));
    assert(overridden.tts_n_timesteps == 6);
    assert(overridden.audio_token_chunk_size == 24);
    assert(overridden.audio_prelook_size == 4);
    assert(overridden.llm_text_chunk_size == 8);
    assert(overridden.tts_max_audio_tokens == 480);
    assert(overridden.duplex_tts_max_audio_tokens == 28);
    assert(overridden.duplex_tts_min_audio_tokens == 24);
    assert(overridden.max_new_speak_tokens_per_chunk == 18);
    assert(overridden.force_listen_count == 2);
    assert(nearly_equal(overridden.listen_prob_scale, 1.2f));

    const OmniGenerationConfig grouped =
        omni_generation_config_from_json({
            {"llm", {
                {"temperature", 0.55},
                {"top_k", 32},
                {"top_p", 0.7},
                {"repetition_penalty", 1.08},
                {"repetition_window", 256},
                {"text_chunk_size", 7},
            }},
            {"tts", {
                {"temperature", 0.6},
                {"top_k", 16},
                {"top_p", 0.9},
                {"repetition_penalty", 1.02},
                {"repetition_window", 12},
                {"min_tokens_to_keep", 4},
                {"n_timesteps", 8},
                {"token2wav_temperature", 0.9},
                {"audio_token_chunk_size", 22},
                {"audio_prelook_size", 2},
                {"max_audio_tokens", 450},
                {"duplex_max_audio_tokens", 27},
                {"duplex_min_audio_tokens", 23},
            }},
        });

    assert(nearly_equal(grouped.llm_sampling.temperature, 0.55f));
    assert(grouped.llm_sampling.top_k == 32);
    assert(nearly_equal(grouped.llm_sampling.top_p, 0.7f));
    assert(nearly_equal(grouped.llm_sampling.repetition_penalty, 1.08f));
    assert(grouped.llm_sampling.repetition_window == 256);
    assert(nearly_equal(grouped.tts_sampling.temperature, 0.6f));
    assert(grouped.tts_sampling.top_k == 16);
    assert(nearly_equal(grouped.tts_sampling.top_p, 0.9f));
    assert(nearly_equal(grouped.tts_sampling.repetition_penalty, 1.02f));
    assert(grouped.tts_sampling.repetition_window == 12);
    assert(grouped.tts_sampling.min_tokens_to_keep == 4);
    assert(grouped.tts_n_timesteps == 8);
    assert(nearly_equal(grouped.token2wav_temperature, 0.9f));
    assert(grouped.audio_token_chunk_size == 22);
    assert(grouped.audio_prelook_size == 2);
    assert(grouped.llm_text_chunk_size == 7);
    assert(grouped.tts_max_audio_tokens == 450);
    assert(grouped.duplex_tts_max_audio_tokens == 27);
    assert(grouped.duplex_tts_min_audio_tokens == 23);

    const std::vector<float> condition = {1.0f, 2.0f};
    const std::vector<float> text_eos_embed = {3.0f, 4.0f};
    const std::vector<float> audio_bos_embed = {5.0f, 6.0f};

    const std::vector<float> final_condition =
        omni_build_tts_condition(condition, text_eos_embed, audio_bos_embed, true);
    assert((final_condition == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

    const std::vector<float> non_final_condition =
        omni_build_tts_condition(condition, text_eos_embed, audio_bos_embed, false);
    assert((non_final_condition == std::vector<float>{1.0f, 2.0f, 5.0f, 6.0f}));

    OmniSamplingConfig tts_sampling = defaults.tts_sampling;
    tts_sampling.top_k = 1;
    assert(omni_effective_tts_top_k(tts_sampling) == 3);

    const OmniTextChunkPlan first_chunk_plan =
        omni_text_chunk_plan(defaults.llm_text_chunk_size, false);
    assert(first_chunk_plan.prefix_tokens == 0);
    assert(first_chunk_plan.condition_tokens == 10);
    assert(first_chunk_plan.generated_tokens == 11);

    const OmniTextChunkPlan continuation_chunk_plan =
        omni_text_chunk_plan(defaults.llm_text_chunk_size, true);
    assert(continuation_chunk_plan.prefix_tokens == 1);
    assert(continuation_chunk_plan.condition_tokens == 9);
    assert(continuation_chunk_plan.generated_tokens == 10);

    const auto llm_sampling_order = omni_python_sampling_order(false);
    assert(llm_sampling_order[0] == OmniSamplingStage::Temperature);
    assert(llm_sampling_order[1] == OmniSamplingStage::TopK);
    assert(llm_sampling_order[2] == OmniSamplingStage::TopP);

    const auto tts_sampling_order = omni_python_sampling_order(true);
    assert(tts_sampling_order[0] == OmniSamplingStage::Temperature);
    assert(tts_sampling_order[1] == OmniSamplingStage::TopP);
    assert(tts_sampling_order[2] == OmniSamplingStage::TopK);

    return 0;
}
