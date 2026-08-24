file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY
    "${TEST_ROOT}/vision"
    "${TEST_ROOT}/audio"
    "${TEST_ROOT}/tts"
    "${TEST_ROOT}/token2wav-gguf"
)

foreach(relative_path IN ITEMS
    MiniCPM-o-4_5-F16.gguf
    MiniCPM-o-4_5-Q8_0.gguf
    MiniCPM-o-4_5-Q4_K_M.gguf
    vision/MiniCPM-o-4_5-vision-F16.gguf
    audio/MiniCPM-o-4_5-audio-F16.gguf
    tts/MiniCPM-o-4_5-tts-F16.gguf
    tts/MiniCPM-o-4_5-projector-F16.gguf
    token2wav-gguf/encoder.gguf
    token2wav-gguf/flow_matching.gguf
    token2wav-gguf/flow_extra.gguf
    token2wav-gguf/hifigan2.gguf
    token2wav-gguf/prompt_cache.gguf
)
    file(WRITE "${TEST_ROOT}/${relative_path}" "test")
endforeach()

execute_process(
    COMMAND "${OMNI_SERVER}"
        --profile auto
        --model-dir "${TEST_ROOT}"
        --autotune-cache "${TEST_ROOT}/missing-autotune.json"
        --print-effective-config
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
    TIMEOUT 20
)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "llama-omni-server profile resolution failed (${result})\nstdout:\n${stdout}\nstderr:\n${stderr}")
endif()

if(NOT stdout MATCHES "profile=auto")
    message(FATAL_ERROR "effective config did not contain profile=auto\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "resolved_profile=full_omni_")
    message(FATAL_ERROR "effective config did not contain a resolved full Omni profile\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "token2wav_model_dir=${TEST_ROOT}/token2wav-gguf")
    message(FATAL_ERROR "effective config did not contain the Token2Wav model directory\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "placement_plan_count=6")
    message(FATAL_ERROR "effective config did not contain all six module placements\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "placement=llm,")
    message(FATAL_ERROR "effective config did not contain the LLM placement\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "placement=token2wav,")
    message(FATAL_ERROR "effective config did not contain the Token2Wav placement\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "autotune_cache=missing")
    message(FATAL_ERROR "normal startup did not report the missing autotune cache\nstdout:\n${stdout}")
endif()

execute_process(
    COMMAND "${OMNI_SERVER}"
        --profile auto
        --model-dir "${TEST_ROOT}"
        --autotune refresh
        --autotune-cache "${TEST_ROOT}/refresh-autotune.json"
    RESULT_VARIABLE refresh_result
    OUTPUT_VARIABLE refresh_stdout
    ERROR_VARIABLE refresh_stderr
    TIMEOUT 20
)

if(refresh_result EQUAL 0)
    message(FATAL_ERROR "autotune refresh unexpectedly succeeded with fake model files\nstdout:\n${refresh_stdout}\nstderr:\n${refresh_stderr}")
endif()
if(refresh_stdout MATCHES "Omni HTTP server starting" OR refresh_stderr MATCHES "Omni HTTP server starting")
    message(FATAL_ERROR "autotune refresh entered the HTTP server path\nstdout:\n${refresh_stdout}\nstderr:\n${refresh_stderr}")
endif()
if(NOT refresh_stderr MATCHES "autotune calibration failed")
    message(FATAL_ERROR "autotune refresh failure was not actionable\nstderr:\n${refresh_stderr}")
endif()

execute_process(
    COMMAND "${OMNI_SERVER}"
        --profile auto
        --model-dir "${TEST_ROOT}"
        --model "${TEST_ROOT}/missing-explicit-model.gguf"
        --print-effective-config
    RESULT_VARIABLE missing_result
    OUTPUT_VARIABLE missing_stdout
    ERROR_VARIABLE missing_stderr
    TIMEOUT 20
)

file(REMOVE_RECURSE "${TEST_ROOT}")

if(missing_result EQUAL 0)
    message(FATAL_ERROR "a missing explicit LLM model was accepted\nstdout:\n${missing_stdout}\nstderr:\n${missing_stderr}")
endif()
if(NOT missing_stderr MATCHES "missing explicit LLM model")
    message(FATAL_ERROR "missing explicit LLM model error was not actionable\nstderr:\n${missing_stderr}")
endif()
