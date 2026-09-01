#include "omni.h"

#include <cassert>
#include <cmath>

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1e-6f;
}

int main() {
    const OmniGenerationConfig simplex =
        omni_python_base_generation_config(false);
    assert(nearly_equal(simplex.llm_sampling.temperature, 0.7f));
    assert(nearly_equal(simplex.llm_sampling.top_p, 0.8f));
    assert(simplex.llm_sampling.top_k == 100);
    assert(nearly_equal(simplex.llm_sampling.repetition_penalty, 1.02f));
    assert(simplex.llm_sampling.repetition_window == 128);
    assert(nearly_equal(simplex.length_penalty, 1.0f));
    const std::vector<llama_token> control_tokens = {10, 11, 12, 13, 14, 15};
    assert(!omni_llm_token_should_update_repetition_history(
        10, control_tokens.data(), control_tokens.size()));
    assert(!omni_llm_token_should_update_repetition_history(
        11, control_tokens.data(), control_tokens.size()));
    assert(!omni_llm_token_should_update_repetition_history(
        13, control_tokens.data(), control_tokens.size()));
    assert(omni_llm_token_should_update_repetition_history(
        100, control_tokens.data(), control_tokens.size()));

    const OmniGenerationConfig duplex =
        omni_python_base_generation_config(true);
    assert(nearly_equal(duplex.llm_sampling.temperature, 0.7f));
    assert(nearly_equal(duplex.llm_sampling.top_p, 0.8f));
    assert(duplex.llm_sampling.top_k == 20);
    assert(nearly_equal(duplex.llm_sampling.repetition_penalty, 1.05f));
    assert(duplex.llm_sampling.repetition_window == 512);
    assert(nearly_equal(duplex.length_penalty, 1.1f));
    assert(duplex.force_listen_count == 0);
    assert(duplex.max_new_speak_tokens_per_chunk == 20);
    assert(!omni_duplex_tts_uses_top_filters());
    assert(omni_duplex_tts_output_token_count_for_budget(26) == 25);
    assert(omni_tts_min_new_tokens_for_chunk(duplex, true, false, 0) == 0);
    assert(omni_tts_min_new_tokens_for_chunk(duplex, true, false, 1) == 26);
    assert(omni_tts_min_new_tokens_for_chunk(duplex, true, true, 1) == 0);

    for (const OmniGenerationConfig * config : {&simplex, &duplex}) {
        assert(config->tts_n_timesteps == 10);
        assert(nearly_equal(config->token2wav_temperature, 1.0f));
        assert(config->audio_token_chunk_size == 25);
        assert(config->audio_prelook_size == 3);
        assert(config->tts_max_audio_tokens == 500);
        assert(config->tts_max_audio_tokens_per_text_token == 30);
        assert(config->tts_min_audio_tokens == 75);
        assert(config->duplex_tts_max_audio_tokens == 26);
        assert(config->duplex_tts_min_audio_tokens == 26);
        assert(config->llm_text_chunk_size == 10);
    }

    const std::vector<float> condition = {1.0f, 2.0f};
    const std::vector<float> text_eos_embed = {3.0f, 4.0f};
    const std::vector<float> audio_bos_embed = {5.0f, 6.0f};

    const std::vector<float> final_condition =
        omni_build_tts_condition(condition, text_eos_embed, audio_bos_embed, true);
    assert((final_condition == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

    const std::vector<float> non_final_condition =
        omni_build_tts_condition(condition, text_eos_embed, audio_bos_embed, false);
    assert((non_final_condition == std::vector<float>{1.0f, 2.0f, 5.0f, 6.0f}));

    OmniSamplingConfig tts_sampling = simplex.tts_sampling;
    tts_sampling.top_k = 1;
    assert(omni_effective_tts_top_k(tts_sampling) == 3);

    const OmniTextChunkPlan first_chunk_plan =
        omni_text_chunk_plan(simplex.llm_text_chunk_size, false);
    assert(first_chunk_plan.prefix_tokens == 0);
    assert(first_chunk_plan.condition_tokens == 10);
    assert(first_chunk_plan.generated_tokens == 11);

    const OmniTextChunkPlan continuation_chunk_plan =
        omni_text_chunk_plan(simplex.llm_text_chunk_size, true);
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
