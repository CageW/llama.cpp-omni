#include "runtime-autotune.h"

#include "ggml-backend.h"
#include "runtime-hash.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <system_error>
#include <thread>
#include <tuple>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#elif defined(__APPLE__)
#    include <dlfcn.h>
#    include <mach-o/dyld.h>
#    include <unistd.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

namespace omni {
namespace {

using json = nlohmann::json;

template <typename T> void put_optional(json & output, const char * key, const std::optional<T> & value) {
    if (value.has_value()) {
        output[key] = *value;
    } else {
        output[key] = nullptr;
    }
}

template <typename T> std::optional<T> get_optional(const json & input, const char * key) {
    const auto & value = input.at(key);
    if (value.is_null()) {
        return std::nullopt;
    }
    return value.get<T>();
}

json serialize_overrides(const runtime_profile_overrides & value) {
    json output;
    put_optional(output, "llm_model", value.llm_model);
    put_optional(output, "llm_device", value.llm_device);
    put_optional(output, "vision_device", value.vision_device);
    put_optional(output, "audio_device", value.audio_device);
    put_optional(output, "tts_device", value.tts_device);
    put_optional(output, "projector_device", value.projector_device);
    put_optional(output, "token2wav_device", value.token2wav_device);
    put_optional(output, "token2wav_threads", value.token2wav_threads);
    put_optional(output, "n_gpu_layers", value.n_gpu_layers);
    put_optional(output, "n_ctx", value.n_ctx);
    put_optional(output, "vpm_batch_encode", value.vpm_batch_encode);
    return output;
}

runtime_profile_overrides deserialize_overrides(const json & input) {
    runtime_profile_overrides output;
    output.llm_model         = get_optional<std::string>(input, "llm_model");
    output.llm_device        = get_optional<std::string>(input, "llm_device");
    output.vision_device     = get_optional<std::string>(input, "vision_device");
    output.audio_device      = get_optional<std::string>(input, "audio_device");
    output.tts_device        = get_optional<std::string>(input, "tts_device");
    output.projector_device  = get_optional<std::string>(input, "projector_device");
    output.token2wav_device  = get_optional<std::string>(input, "token2wav_device");
    output.token2wav_threads = get_optional<int32_t>(input, "token2wav_threads");
    output.n_gpu_layers      = get_optional<int32_t>(input, "n_gpu_layers");
    output.n_ctx             = get_optional<int32_t>(input, "n_ctx");
    output.vpm_batch_encode  = get_optional<bool>(input, "vpm_batch_encode");
    return output;
}

json serialize_build(const autotune_build_identity & value) {
    return {
        { "number", value.build_number },
        { "commit", value.commit       },
        { "target", value.target       },
    };
}

autotune_build_identity deserialize_build(const json & input) {
    return {
        input.at("number").get<int32_t>(),
        input.at("commit").get<std::string>(),
        input.at("target").get<std::string>(),
    };
}

json serialize_runtime_identity(const autotune_runtime_identity & value) {
    json libraries = json::array();
    for (const auto & library : value.backend_libraries) {
        libraries.push_back({
            { "path",          library.path          },
            { "size",          library.size          },
            { "modified_time", library.modified_time },
            { "sha256",        library.sha256        },
        });
    }
    return {
        { "executable_sha256",    value.executable_sha256    },
        { "backend_libraries",    std::move(libraries)       },
        { "cuda_driver_version",  value.cuda_driver_version  },
        { "cuda_runtime_version", value.cuda_runtime_version },
    };
}

autotune_runtime_identity deserialize_runtime_identity(const json & input) {
    autotune_runtime_identity output;
    output.executable_sha256    = input.at("executable_sha256").get<std::string>();
    output.cuda_driver_version  = input.at("cuda_driver_version").get<std::string>();
    output.cuda_runtime_version = input.at("cuda_runtime_version").get<std::string>();
    for (const auto & library : input.at("backend_libraries")) {
        output.backend_libraries.push_back({
            library.at("path").get<std::string>(),
            library.at("size").get<uint64_t>(),
            library.at("modified_time").get<int64_t>(),
            library.at("sha256").get<std::string>(),
        });
    }
    return output;
}

json serialize_fingerprint(const autotune_fingerprint & value) {
    json devices = json::array();
    for (const auto & device : value.devices) {
        devices.push_back({
            { "name",           device.name           },
            { "description",    device.description    },
            { "is_accelerator", device.is_accelerator },
            { "total_memory",   device.total_memory   },
            { "type",           device.type           },
        });
    }

    json files = json::array();
    for (const auto & file : value.files) {
        files.push_back({
            { "path",          file.path          },
            { "size",          file.size          },
            { "modified_time", file.modified_time },
        });
    }

    return {
        { "schema_version",    value.schema_version                               },
        { "scoring_policy",    value.scoring_policy                               },
        { "build",             serialize_build(value.build)                       },
        { "requested_profile", value.requested_profile                            },
        { "overrides",         serialize_overrides(value.overrides)               },
        { "runtime_inputs",    value.runtime_inputs                               },
        { "runtime_identity",  serialize_runtime_identity(value.runtime_identity) },
        { "devices",           std::move(devices)                                 },
        { "files",             std::move(files)                                   },
    };
}

autotune_fingerprint deserialize_fingerprint(const json & input) {
    autotune_fingerprint output;
    output.schema_version    = input.at("schema_version").get<int32_t>();
    output.scoring_policy    = input.at("scoring_policy").get<int32_t>();
    output.build             = deserialize_build(input.at("build"));
    output.requested_profile = input.at("requested_profile").get<std::string>();
    output.overrides         = deserialize_overrides(input.at("overrides"));
    output.runtime_inputs    = input.at("runtime_inputs").get<std::map<std::string, std::string>>();
    if (input.contains("runtime_identity")) {
        output.runtime_identity = deserialize_runtime_identity(input.at("runtime_identity"));
    }

    for (const auto & device : input.at("devices")) {
        output.devices.push_back({
            device.at("name").get<std::string>(),
            device.at("description").get<std::string>(),
            device.at("is_accelerator").get<bool>(),
            device.at("total_memory").get<uint64_t>(),
            device.at("type").get<std::string>(),
        });
    }
    for (const auto & file : input.at("files")) {
        output.files.push_back({
            file.at("path").get<std::string>(),
            file.at("size").get<uint64_t>(),
            file.at("modified_time").get<int64_t>(),
        });
    }
    return output;
}

json serialize_placement(const module_placement & value) {
    return {
        { "module",    value.module    },
        { "model",     value.model     },
        { "precision", value.precision },
        { "device",    value.device    },
        { "backend",   value.backend   },
        { "execution", value.execution },
        { "status",    value.status    },
    };
}

module_placement deserialize_placement(const json & input) {
    return {
        input.at("module").get<std::string>(),    input.at("model").get<std::string>(),
        input.at("precision").get<std::string>(), input.at("device").get<std::string>(),
        input.at("backend").get<std::string>(),   input.at("execution").get<std::string>(),
        input.at("status").get<std::string>(),
    };
}

json serialize_config(const effective_runtime_config & value) {
    json placements = json::array();
    for (const auto & placement : value.placements) {
        placements.push_back(serialize_placement(placement));
    }

    return {
        { "requested_profile",               value.requested_profile               },
        { "resolved_profile",                value.resolved_profile                },
        { "reason",                          value.reason                          },
        { "primary_device_name",             value.primary_device_name             },
        { "primary_device_description",      value.primary_device_description      },
        { "primary_device_free_memory",      value.primary_device_free_memory      },
        { "primary_device_total_memory",     value.primary_device_total_memory     },
        { "estimated_required_memory",       value.estimated_required_memory       },
        { "llm_model",                       value.llm_model                       },
        { "llm_quantization",                value.llm_quantization                },
        { "llm_device",                      value.llm_device                      },
        { "vision_model",                    value.vision_model                    },
        { "vision_device",                   value.vision_device                   },
        { "audio_model",                     value.audio_model                     },
        { "audio_device",                    value.audio_device                    },
        { "tts_model",                       value.tts_model                       },
        { "tts_device",                      value.tts_device                      },
        { "projector_model",                 value.projector_model                 },
        { "token2wav_model_dir",             value.token2wav_model_dir             },
        { "n_gpu_layers",                    value.n_gpu_layers                    },
        { "tts_gpu_layers",                  value.tts_gpu_layers                  },
        { "token2wav_device",                value.token2wav_device                },
        { "token2wav_threads",               value.token2wav_threads               },
        { "n_ctx",                           value.n_ctx                           },
        { "duplex_mode",                     value.duplex_mode                     },
        { "async_mode",                      value.async_mode                      },
        { "vpm_batch_encode",                value.vpm_batch_encode                },
        { "autotune_cache_status",           value.autotune_cache_status           },
        { "autotune_cache_reason",           value.autotune_cache_reason           },
        { "autotune_selected_candidate",     value.autotune_selected_candidate     },
        { "output_consistent_with_baseline", value.output_consistent_with_baseline },
        { "audio_semantics_verified",        value.audio_semantics_verified        },
        { "placements",                      std::move(placements)                 },
        { "fallback_reasons",                value.fallback_reasons                },
    };
}

effective_runtime_config deserialize_config(const json & input) {
    effective_runtime_config output;
    output.requested_profile               = input.at("requested_profile").get<std::string>();
    output.resolved_profile                = input.at("resolved_profile").get<std::string>();
    output.reason                          = input.at("reason").get<std::string>();
    output.primary_device_name             = input.at("primary_device_name").get<std::string>();
    output.primary_device_description      = input.at("primary_device_description").get<std::string>();
    output.primary_device_free_memory      = input.at("primary_device_free_memory").get<uint64_t>();
    output.primary_device_total_memory     = input.at("primary_device_total_memory").get<uint64_t>();
    output.estimated_required_memory       = input.at("estimated_required_memory").get<uint64_t>();
    output.llm_model                       = input.at("llm_model").get<std::string>();
    output.llm_quantization                = input.at("llm_quantization").get<std::string>();
    output.llm_device                      = input.at("llm_device").get<std::string>();
    output.vision_model                    = input.at("vision_model").get<std::string>();
    output.vision_device                   = input.at("vision_device").get<std::string>();
    output.audio_model                     = input.at("audio_model").get<std::string>();
    output.audio_device                    = input.at("audio_device").get<std::string>();
    output.tts_model                       = input.at("tts_model").get<std::string>();
    output.tts_device                      = input.at("tts_device").get<std::string>();
    output.projector_model                 = input.at("projector_model").get<std::string>();
    output.token2wav_model_dir             = input.at("token2wav_model_dir").get<std::string>();
    output.n_gpu_layers                    = input.at("n_gpu_layers").get<int32_t>();
    output.tts_gpu_layers                  = input.at("tts_gpu_layers").get<int32_t>();
    output.token2wav_device                = input.at("token2wav_device").get<std::string>();
    output.token2wav_threads               = input.at("token2wav_threads").get<int32_t>();
    output.n_ctx                           = input.at("n_ctx").get<int32_t>();
    output.duplex_mode                     = input.at("duplex_mode").get<bool>();
    output.async_mode                      = input.at("async_mode").get<bool>();
    output.vpm_batch_encode                = input.at("vpm_batch_encode").get<bool>();
    output.autotune_cache_status           = input.at("autotune_cache_status").get<std::string>();
    output.autotune_cache_reason           = input.at("autotune_cache_reason").get<std::string>();
    output.autotune_selected_candidate     = input.at("autotune_selected_candidate").get<std::string>();
    output.output_consistent_with_baseline = input.at("output_consistent_with_baseline").get<bool>();
    output.audio_semantics_verified        = input.at("audio_semantics_verified").get<bool>();
    output.fallback_reasons                = input.at("fallback_reasons").get<std::vector<std::string>>();
    for (const auto & placement : input.at("placements")) {
        output.placements.push_back(deserialize_placement(placement));
    }
    return output;
}

json serialize_metrics(const autotune_metrics & value) {
    return {
        { "frame_total_p95_ms", value.frame_total_p95_ms },
        { "encoder_p95_ms",     value.encoder_p95_ms     },
        { "decode_p95_ms",      value.decode_p95_ms      },
        { "ttft_p95_ms",        value.ttft_p95_ms        },
        { "first_audio_p95_ms", value.first_audio_p95_ms },
        { "tts_rtf",            value.tts_rtf            },
        { "frame_count",        value.frame_count        },
        { "speak_count",        value.speak_count        },
        { "final_audio_count",  value.final_audio_count  },
        { "audio_sample_count", value.audio_sample_count },
        { "timeout_count",      value.timeout_count      },
    };
}

autotune_metrics deserialize_metrics(const json & input) {
    autotune_metrics output;
    output.frame_total_p95_ms = input.at("frame_total_p95_ms").get<double>();
    output.encoder_p95_ms     = input.at("encoder_p95_ms").get<double>();
    output.decode_p95_ms      = input.at("decode_p95_ms").get<double>();
    output.ttft_p95_ms        = input.at("ttft_p95_ms").get<double>();
    output.first_audio_p95_ms = input.at("first_audio_p95_ms").get<double>();
    output.tts_rtf            = input.at("tts_rtf").get<double>();
    output.frame_count        = input.at("frame_count").get<int32_t>();
    output.speak_count        = input.at("speak_count").get<int32_t>();
    output.final_audio_count  = input.at("final_audio_count").get<int32_t>();
    output.audio_sample_count = input.at("audio_sample_count").get<int64_t>();
    output.timeout_count      = input.at("timeout_count").get<int32_t>();
    return output;
}

json serialize_memory_samples(const std::vector<autotune_memory_sample> & samples) {
    json output = json::array();
    for (const auto & sample : samples) {
        output.push_back({
            { "checkpoint",             sample.checkpoint             },
            { "free_memory_bytes",      sample.free_memory_bytes      },
            { "used_delta_bytes",       sample.used_delta_bytes       },
            { "incremental_used_bytes", sample.incremental_used_bytes },
        });
    }
    return output;
}

std::vector<autotune_memory_sample> deserialize_memory_samples(const json & input) {
    std::vector<autotune_memory_sample> output;
    for (const auto & sample : input) {
        output.push_back({
            sample.at("checkpoint").get<std::string>(),
            sample.at("free_memory_bytes").get<std::map<std::string, uint64_t>>(),
            sample.at("used_delta_bytes").get<std::map<std::string, uint64_t>>(),
            sample.at("incremental_used_bytes").get<std::map<std::string, uint64_t>>(),
        });
    }
    return output;
}

json serialize_rounds(const std::vector<autotune_round_record> & rounds) {
    json output = json::array();
    for (const auto & round : rounds) {
        output.push_back({
            { "round_index",         round.round_index                              },
            { "metrics",             serialize_metrics(round.metrics)               },
            { "memory_samples",      serialize_memory_samples(round.memory_samples) },
            { "device_memory_peaks", round.device_memory_peaks                      },
            { "response_sha256",     round.response_sha256                          },
            { "tts_tokens_sha256",   round.tts_tokens_sha256                        },
            { "tts_token_count",     round.tts_token_count                          },
            { "modules_initialized", round.modules_initialized                      },
            { "samples_finite",      round.samples_finite                           },
            { "valid",               round.valid                                    },
            { "execution_error",     round.execution_error                          },
            { "rejection_reason",    round.rejection_reason                         },
            { "score",               round.score                                    },
        });
    }
    return output;
}

json serialize_round(const autotune_round_record & round) {
    return serialize_rounds({ round }).front();
}

std::vector<autotune_round_record> deserialize_rounds(const json & input) {
    std::vector<autotune_round_record> output;
    for (const auto & round : input) {
        autotune_round_record value;
        value.round_index         = round.at("round_index").get<int32_t>();
        value.metrics             = deserialize_metrics(round.at("metrics"));
        value.memory_samples      = deserialize_memory_samples(round.at("memory_samples"));
        value.device_memory_peaks = round.at("device_memory_peaks").get<std::map<std::string, uint64_t>>();
        value.response_sha256     = round.at("response_sha256").get<std::string>();
        value.tts_tokens_sha256   = round.at("tts_tokens_sha256").get<std::string>();
        value.tts_token_count     = round.at("tts_token_count").get<int64_t>();
        value.modules_initialized = round.at("modules_initialized").get<bool>();
        value.samples_finite      = round.at("samples_finite").get<bool>();
        value.valid               = round.at("valid").get<bool>();
        value.execution_error     = round.at("execution_error").get<std::string>();
        value.rejection_reason    = round.at("rejection_reason").get<std::string>();
        value.score               = round.at("score").get<double>();
        output.push_back(std::move(value));
    }
    return output;
}

autotune_round_record deserialize_round(const json & input) {
    return deserialize_rounds(json::array({ input })).front();
}

json serialize_candidate(const autotune_candidate_record & value) {
    return {
        { "id",                         value.id                                       },
        { "stage",                      value.stage                                    },
        { "config",                     serialize_config(value.config)                 },
        { "metrics",                    serialize_metrics(value.metrics)               },
        { "memory_estimates",           value.memory_estimates                         },
        { "rounds",                     serialize_rounds(value.rounds)                 },
        { "memory_samples",             serialize_memory_samples(value.memory_samples) },
        { "device_memory_peaks",        value.device_memory_peaks                      },
        { "response_sha256",            value.response_sha256                          },
        { "tts_tokens_sha256",          value.tts_tokens_sha256                        },
        { "tts_token_count",            value.tts_token_count                          },
        { "selectable",                 value.selectable                               },
        { "advisory",                   value.advisory                                 },
        { "screening_performed",        value.screening_performed                      },
        { "screening_rejected",         value.screening_rejected                       },
        { "screening",                  serialize_round(value.screening)               },
        { "output_consistency_checked", value.output_consistency_checked               },
        { "output_matches_baseline",    value.output_matches_baseline                  },
        { "modules_initialized",        value.modules_initialized                      },
        { "samples_finite",             value.samples_finite                           },
        { "valid",                      value.valid                                    },
        { "rejection_reason",           value.rejection_reason                         },
        { "score",                      value.score                                    },
    };
}

autotune_candidate_record deserialize_candidate(const json & input) {
    autotune_candidate_record output;
    output.id                         = input.at("id").get<std::string>();
    output.stage                      = input.at("stage").get<std::string>();
    output.config                     = deserialize_config(input.at("config"));
    output.metrics                    = deserialize_metrics(input.at("metrics"));
    output.memory_estimates           = input.at("memory_estimates").get<std::map<std::string, uint64_t>>();
    output.rounds                     = deserialize_rounds(input.at("rounds"));
    output.memory_samples             = deserialize_memory_samples(input.at("memory_samples"));
    output.device_memory_peaks        = input.at("device_memory_peaks").get<std::map<std::string, uint64_t>>();
    output.response_sha256            = input.at("response_sha256").get<std::string>();
    output.tts_tokens_sha256          = input.at("tts_tokens_sha256").get<std::string>();
    output.tts_token_count            = input.at("tts_token_count").get<int64_t>();
    output.selectable                 = input.at("selectable").get<bool>();
    output.advisory                   = input.at("advisory").get<bool>();
    output.screening_performed        = input.at("screening_performed").get<bool>();
    output.screening_rejected         = input.at("screening_rejected").get<bool>();
    output.screening                  = deserialize_round(input.at("screening"));
    output.output_consistency_checked = input.at("output_consistency_checked").get<bool>();
    output.output_matches_baseline    = input.at("output_matches_baseline").get<bool>();
    output.modules_initialized        = input.at("modules_initialized").get<bool>();
    output.samples_finite             = input.at("samples_finite").get<bool>();
    output.valid                      = input.at("valid").get<bool>();
    output.rejection_reason           = input.at("rejection_reason").get<std::string>();
    output.score                      = input.at("score").get<double>();
    return output;
}

json serialize_cache(const autotune_cache_record & value) {
    json candidates = json::array();
    for (const auto & candidate : value.candidates) {
        candidates.push_back(serialize_candidate(candidate));
    }
    return {
        { "schema_version",           value.schema_version                     },
        { "fingerprint",              serialize_fingerprint(value.fingerprint) },
        { "selected_candidate",       value.selected_candidate                 },
        { "selection_reason",         value.selection_reason                   },
        { "candidates",               std::move(candidates)                    },
        { "skipped_reasons",          value.skipped_reasons                    },
        { "memory_estimate_inputs",   value.memory_estimate_inputs             },
        { "audio_semantics_verified", value.audio_semantics_verified           },
        { "created_at",               value.created_at                         },
    };
}

autotune_cache_record deserialize_cache(const json & input) {
    autotune_cache_record output;
    output.schema_version           = input.at("schema_version").get<int32_t>();
    output.fingerprint              = deserialize_fingerprint(input.at("fingerprint"));
    output.selected_candidate       = input.at("selected_candidate").get<std::string>();
    output.selection_reason         = input.at("selection_reason").get<std::string>();
    output.skipped_reasons          = input.at("skipped_reasons").get<std::vector<std::string>>();
    output.memory_estimate_inputs   = input.at("memory_estimate_inputs").get<std::map<std::string, uint64_t>>();
    output.audio_semantics_verified = input.at("audio_semantics_verified").get<bool>();
    output.created_at               = input.at("created_at").get<std::string>();
    for (const auto & candidate : input.at("candidates")) {
        output.candidates.push_back(deserialize_candidate(candidate));
    }
    return output;
}

bool overrides_equal(const runtime_profile_overrides & left, const runtime_profile_overrides & right) {
    return left.llm_model == right.llm_model && left.llm_device == right.llm_device &&
           left.vision_device == right.vision_device && left.audio_device == right.audio_device &&
           left.tts_device == right.tts_device && left.projector_device == right.projector_device &&
           left.token2wav_device == right.token2wav_device && left.n_gpu_layers == right.n_gpu_layers &&
           left.token2wav_threads == right.token2wav_threads && left.n_ctx == right.n_ctx &&
           left.vpm_batch_encode == right.vpm_batch_encode;
}

bool replace_file(const std::filesystem::path & source,
                  const std::filesystem::path & destination,
                  std::string &                 error) {
#ifdef _WIN32
    if (MoveFileExW(source.wstring().c_str(), destination.wstring().c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
        error = "failed to replace autotune cache: Windows error " + std::to_string(GetLastError());
        return false;
    }
    return true;
#else
    std::error_code filesystem_error;
    std::filesystem::rename(source, destination, filesystem_error);
    if (filesystem_error) {
        error = "failed to replace autotune cache: " + filesystem_error.message();
        return false;
    }
    return true;
#endif
}

std::vector<const runtime_device *> sorted_accelerators(const hardware_snapshot & hardware) {
    std::vector<const runtime_device *> output;
    for (const auto & device : hardware.devices) {
        if (device.is_accelerator) {
            output.push_back(&device);
        }
    }
    std::sort(output.begin(), output.end(), [](const auto * left, const auto * right) {
        if (left->free_memory != right->free_memory) {
            return left->free_memory > right->free_memory;
        }
        return left->name < right->name;
    });
    return output;
}

const runtime_device * find_runtime_device(const hardware_snapshot & hardware, const std::string & name) {
    for (const auto & device : hardware.devices) {
        if (device.name == name) {
            return &device;
        }
    }
    return nullptr;
}

bool module_has_device_override(const std::string & module, const runtime_profile_overrides & overrides) {
    if (module == "llm") {
        return overrides.llm_device.has_value();
    }
    if (module == "vision") {
        return overrides.vision_device.has_value();
    }
    if (module == "audio") {
        return overrides.audio_device.has_value();
    }
    if (module == "tts") {
        return overrides.tts_device.has_value();
    }
    if (module == "projector") {
        return overrides.projector_device.has_value();
    }
    if (module == "token2wav") {
        return overrides.token2wav_device.has_value();
    }
    return false;
}

std::string token2wav_device_name(const std::string & backend_device) {
    std::string digits;
    for (auto iterator = backend_device.rbegin();
         iterator != backend_device.rend() && std::isdigit(static_cast<unsigned char>(*iterator)); ++iterator) {
        digits.push_back(*iterator);
    }
    if (digits.empty()) {
        return "gpu:0";
    }
    std::reverse(digits.begin(), digits.end());
    return "gpu:" + digits;
}

void assign_candidate_device(effective_runtime_config &        config,
                             const std::string &               module,
                             const std::string &               device,
                             const runtime_profile_overrides & overrides) {
    if (module_has_device_override(module, overrides)) {
        return;
    }
    for (auto & placement : config.placements) {
        if (placement.module != module) {
            continue;
        }
        placement.device  = device;
        placement.backend = device == "cpu" ? "cpu" : "gpu";
        break;
    }
    if (module == "llm") {
        config.llm_device = device;
    } else if (module == "vision") {
        config.vision_device = device;
    } else if (module == "audio") {
        config.audio_device = device;
    } else if (module == "tts") {
        config.tts_device = device;
    } else if (module == "token2wav") {
        config.token2wav_device = device == "cpu" ? "cpu" : token2wav_device_name(device);
    }
}

effective_runtime_config make_candidate_config(const effective_runtime_config &  base_config,
                                               const std::string &               id,
                                               const std::string &               primary,
                                               const std::string &               encoder_device,
                                               const std::string &               tts_device,
                                               const std::string &               projector_device,
                                               const std::string &               token2wav_device,
                                               const runtime_profile_overrides & overrides) {
    effective_runtime_config output = base_config;
    output.resolved_profile         = "autotune_" + id;
    if (id == "consolidated") {
        output.reason = "calibration candidate places every module on the primary accelerator";
    } else if (id == "encoder_split") {
        output.reason = "calibration candidate places Vision, Audio, and Projector on the secondary accelerator";
    } else if (id == "encoder_split_local_projector") {
        output.reason = "calibration candidate places Vision and Audio on the secondary accelerator";
    } else if (id == "token2wav_split") {
        output.reason =
            "calibration candidate places Vision, Audio, Projector, and Token2Wav on the secondary accelerator";
    } else if (id == "output_pipeline_split") {
        output.reason = "calibration candidate places every output module on the secondary accelerator";
    } else {
        output.reason =
            "calibration candidate places Vision and Audio on the secondary accelerator and Token2Wav on CPU";
    }
    assign_candidate_device(output, "llm", primary, overrides);
    assign_candidate_device(output, "vision", encoder_device, overrides);
    assign_candidate_device(output, "audio", encoder_device, overrides);
    assign_candidate_device(output, "tts", tts_device, overrides);
    assign_candidate_device(output, "projector", projector_device, overrides);
    assign_candidate_device(output, "token2wav", token2wav_device, overrides);
    return output;
}

bool placement_matches(const effective_runtime_config & left, const effective_runtime_config & right) {
    for (const auto & left_placement : left.placements) {
        const auto match = std::find_if(right.placements.begin(), right.placements.end(), [&](const auto & placement) {
            return placement.module == left_placement.module && placement.device == left_placement.device &&
                   placement.status == left_placement.status;
        });
        if (match == right.placements.end()) {
            return false;
        }
    }
    return left.placements.size() == right.placements.size();
}

std::string candidate_config_key(const effective_runtime_config & config) {
    std::vector<std::string> placements;
    for (const auto & placement : config.placements) {
        placements.push_back(placement.module + "=" + placement.device + ":" + placement.status);
    }
    std::sort(placements.begin(), placements.end());
    std::ostringstream output;
    output << config.llm_model << '|' << config.llm_quantization << '|' << config.n_gpu_layers << '|' << config.n_ctx
           << '|' << config.token2wav_threads;
    for (const auto & placement : placements) {
        output << '|' << placement;
    }
    return output.str();
}

const module_placement * find_module_placement(const effective_runtime_config & config, const std::string & module) {
    for (const auto & placement : config.placements) {
        if (placement.module == module) {
            return &placement;
        }
    }
    return nullptr;
}

uint64_t scaled_model_bytes(uint64_t bytes) {
    return static_cast<uint64_t>(std::ceil(static_cast<double>(bytes) * 1.10));
}

bool model_file_estimate(const std::string & path, uint64_t overhead, uint64_t & estimate) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) {
        return false;
    }
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        return false;
    }
    estimate = scaled_model_bytes(size) + overhead;
    return true;
}

bool model_directory_estimate(const std::string & path, uint64_t overhead, uint64_t & estimate) {
    std::error_code error;
    if (!std::filesystem::is_directory(path, error) || error) {
        return false;
    }
    uint64_t bytes = 0;
    for (std::filesystem::recursive_directory_iterator iterator(path, error), end; iterator != end && !error;
         iterator.increment(error)) {
        if (iterator->is_regular_file(error) && !error) {
            bytes += iterator->file_size(error);
        }
    }
    if (error) {
        return false;
    }
    estimate = scaled_model_bytes(bytes) + overhead;
    return true;
}

bool estimate_candidate_memory(const effective_runtime_config &        config,
                               const std::map<std::string, uint64_t> & previous_measured_module_bytes,
                               std::map<std::string, uint64_t> &       estimates) {
    static constexpr uint64_t MiB = 1024ULL * 1024ULL;
    static constexpr uint64_t GiB = 1024ULL * MiB;

    struct module_input {
        const char *        module;
        const std::string * path;
        uint64_t            overhead;
        bool                directory;
    };

    const module_input modules[] = {
        { "llm",       &config.llm_model,           2 * GiB,   false },
        { "vision",    &config.vision_model,        1 * GiB,   false },
        { "audio",     &config.audio_model,         512 * MiB, false },
        { "tts",       &config.tts_model,           1 * GiB,   false },
        { "projector", &config.projector_model,     256 * MiB, false },
        { "token2wav", &config.token2wav_model_dir, 1 * GiB,   true  },
    };
    estimates.clear();
    for (const auto & module : modules) {
        const auto * placement = find_module_placement(config, module.module);
        if (placement == nullptr) {
            return false;
        }
        uint64_t   estimate  = 0;
        const bool available = module.directory ? model_directory_estimate(*module.path, module.overhead, estimate) :
                                                  model_file_estimate(*module.path, module.overhead, estimate);
        if (!available) {
            estimates.clear();
            return false;
        }
        const auto measured = previous_measured_module_bytes.find(module.module);
        if (measured != previous_measured_module_bytes.end()) {
            estimate = std::max(estimate, scaled_model_bytes(measured->second));
        }
        estimates[placement->device] += estimate;
    }
    return true;
}

bool candidate_fits_memory(const autotune_candidate_record & candidate,
                           const hardware_snapshot &         hardware,
                           std::string &                     reason) {
    for (const auto & entry : candidate.memory_estimates) {
        if (entry.first == "cpu") {
            continue;
        }
        const auto * device = find_runtime_device(hardware, entry.first);
        if (device == nullptr || !device->is_accelerator) {
            reason = candidate.id + " references unavailable accelerator " + entry.first;
            return false;
        }
        const uint64_t usable = device->free_memory - device->free_memory / 10;
        if (entry.second > usable) {
            reason = candidate.id + " requires " + std::to_string(entry.second) + " bytes on " + entry.first +
                     ", exceeding the 90% free-memory budget of " + std::to_string(usable) + " bytes";
            return false;
        }
    }
    return true;
}

bool metrics_are_finite(const autotune_metrics & metrics) {
    return std::isfinite(metrics.frame_total_p95_ms) && std::isfinite(metrics.encoder_p95_ms) &&
           std::isfinite(metrics.decode_p95_ms) && std::isfinite(metrics.ttft_p95_ms) &&
           std::isfinite(metrics.first_audio_p95_ms) && std::isfinite(metrics.tts_rtf);
}

bool is_sha256_hex(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
           });
}

double policy_two_score(const autotune_metrics & metrics) {
    return 0.20 * metrics.encoder_p95_ms / 1000.0 + 0.15 * metrics.ttft_p95_ms / 250.0 +
           0.25 * metrics.first_audio_p95_ms / 1000.0 + 0.40 * metrics.tts_rtf;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

double median_plus_mad(const std::vector<double> & values) {
    const double        center = median(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const double value : values) {
        deviations.push_back(std::abs(value - center));
    }
    return center + median(std::move(deviations));
}

bool configs_match_for_cache(const effective_runtime_config & cached, const effective_runtime_config & expected) {
    auto cached_json   = serialize_config(cached);
    auto expected_json = serialize_config(expected);
    for (const char * dynamic_field : {
             "primary_device_free_memory",
             "autotune_cache_status",
             "autotune_cache_reason",
             "autotune_selected_candidate",
             "output_consistent_with_baseline",
             "audio_semantics_verified",
         }) {
        cached_json.erase(dynamic_field);
        expected_json.erase(dynamic_field);
    }
    return cached_json == expected_json;
}

bool scalars_match(double left, double right) {
    return std::isfinite(left) && std::isfinite(right) && std::abs(left - right) <= 1e-12;
}

bool metrics_match(const autotune_metrics & left, const autotune_metrics & right) {
    return scalars_match(left.frame_total_p95_ms, right.frame_total_p95_ms) &&
           scalars_match(left.encoder_p95_ms, right.encoder_p95_ms) &&
           scalars_match(left.decode_p95_ms, right.decode_p95_ms) &&
           scalars_match(left.ttft_p95_ms, right.ttft_p95_ms) &&
           scalars_match(left.first_audio_p95_ms, right.first_audio_p95_ms) &&
           scalars_match(left.tts_rtf, right.tts_rtf) && left.frame_count == right.frame_count &&
           left.speak_count == right.speak_count && left.final_audio_count == right.final_audio_count &&
           left.audio_sample_count == right.audio_sample_count && left.timeout_count == right.timeout_count;
}

autotune_calibration_observation observation_from_round(const autotune_round_record & round) {
    autotune_calibration_observation output;
    output.metrics             = round.metrics;
    output.memory_samples      = round.memory_samples;
    output.response_sha256     = round.response_sha256;
    output.tts_tokens_sha256   = round.tts_tokens_sha256;
    output.tts_token_count     = round.tts_token_count;
    output.modules_initialized = round.modules_initialized;
    output.samples_finite      = round.samples_finite;
    output.execution_error     = round.execution_error;
    return output;
}

bool round_evaluation_matches(const autotune_round_record & cached, const autotune_round_record & evaluated) {
    return cached.valid == evaluated.valid && cached.rejection_reason == evaluated.rejection_reason &&
           scalars_match(cached.score, evaluated.score) && cached.device_memory_peaks == evaluated.device_memory_peaks;
}

std::string executable_path(std::string & error) {
#ifdef _WIN32
    std::vector<char> buffer(32768);
    const DWORD       length = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        error = "cannot resolve current executable path";
        return {};
    }
    return std::string(buffer.data(), length);
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        error = "cannot resolve current executable path";
        return {};
    }
    return buffer.data();
#else
    std::vector<char> buffer(4096);
    const ssize_t     length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length < 0) {
        error = "cannot resolve current executable path";
        return {};
    }
    buffer[static_cast<size_t>(length)] = '\0';
    return buffer.data();
#endif
}

std::string library_path_for_address(void * address) {
    if (address == nullptr) {
        return {};
    }
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(address), &module)) {
        return {};
    }
    std::vector<char> buffer(32768);
    const DWORD       length = GetModuleFileNameA(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    return length > 0 && length < buffer.size() ? std::string(buffer.data(), length) : std::string();
#else
    Dl_info info{};
    return dladdr(address, &info) != 0 && info.dli_fname != nullptr ? info.dli_fname : std::string();
#endif
}

bool make_backend_identity(const std::string & path, autotune_backend_identity & output, std::string & error) {
    std::error_code filesystem_error;
    const auto      canonical = std::filesystem::weakly_canonical(path, filesystem_error);
    if (filesystem_error) {
        error = "cannot canonicalize GGML backend library " + path + ": " + filesystem_error.message();
        return false;
    }
    const auto size = std::filesystem::file_size(canonical, filesystem_error);
    if (filesystem_error) {
        error = "cannot read GGML backend library size " + canonical.string() + ": " + filesystem_error.message();
        return false;
    }
    const auto modified = std::filesystem::last_write_time(canonical, filesystem_error);
    if (filesystem_error) {
        error = "cannot read GGML backend library timestamp " + canonical.string() + ": " + filesystem_error.message();
        return false;
    }
    std::string digest;
    if (!sha256_file(canonical.string(), digest, error)) {
        return false;
    }
    output = { canonical.string(), size, static_cast<int64_t>(modified.time_since_epoch().count()), digest };
    return true;
}

std::string format_cuda_version(int version) {
    if (version <= 0) {
        return "unavailable";
    }
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10) + " (" +
           std::to_string(version) + ")";
}

std::string query_cuda_version(const char * symbol_name, const std::vector<const char *> & libraries) {
    using version_function = int (*)(int *);
    void * symbol          = nullptr;
#ifdef _WIN32
    HMODULE loaded = nullptr;
    symbol         = reinterpret_cast<void *>(GetProcAddress(GetModuleHandleA("nvcuda.dll"), symbol_name));
    for (const char * library : libraries) {
        if (symbol != nullptr) {
            break;
        }
        loaded = LoadLibraryA(library);
        if (loaded != nullptr) {
            symbol = reinterpret_cast<void *>(GetProcAddress(loaded, symbol_name));
        }
    }
#else
    void * loaded = nullptr;
    symbol        = dlsym(RTLD_DEFAULT, symbol_name);
    for (const char * library : libraries) {
        if (symbol != nullptr) {
            break;
        }
        loaded = dlopen(library, RTLD_LAZY | RTLD_LOCAL);
        if (loaded != nullptr) {
            symbol = dlsym(loaded, symbol_name);
        }
    }
#endif
    int version = 0;
    if (symbol != nullptr) {
        reinterpret_cast<version_function>(symbol)(&version);
    }
#ifdef _WIN32
    if (loaded != nullptr) {
        FreeLibrary(loaded);
    }
#else
    if (loaded != nullptr) {
        dlclose(loaded);
    }
#endif
    return format_cuda_version(version);
}

}  // namespace

bool operator==(const autotune_build_identity & left, const autotune_build_identity & right) {
    return left.build_number == right.build_number && left.commit == right.commit && left.target == right.target;
}

bool operator==(const autotune_file_fingerprint & left, const autotune_file_fingerprint & right) {
    return left.path == right.path && left.size == right.size && left.modified_time == right.modified_time;
}

bool operator==(const autotune_backend_identity & left, const autotune_backend_identity & right) {
    return left.path == right.path && left.size == right.size && left.modified_time == right.modified_time &&
           left.sha256 == right.sha256;
}

bool operator==(const autotune_runtime_identity & left, const autotune_runtime_identity & right) {
    return left.executable_sha256 == right.executable_sha256 && left.backend_libraries == right.backend_libraries &&
           left.cuda_driver_version == right.cuda_driver_version &&
           left.cuda_runtime_version == right.cuda_runtime_version;
}

bool operator==(const autotune_device_fingerprint & left, const autotune_device_fingerprint & right) {
    return left.name == right.name && left.description == right.description &&
           left.is_accelerator == right.is_accelerator && left.total_memory == right.total_memory &&
           left.type == right.type;
}

bool operator==(const autotune_fingerprint & left, const autotune_fingerprint & right) {
    return left.schema_version == right.schema_version && left.scoring_policy == right.scoring_policy &&
           left.build == right.build && left.requested_profile == right.requested_profile &&
           overrides_equal(left.overrides, right.overrides) && left.runtime_inputs == right.runtime_inputs &&
           left.runtime_identity == right.runtime_identity && left.devices == right.devices &&
           left.files == right.files;
}

bool operator!=(const autotune_fingerprint & left, const autotune_fingerprint & right) {
    return !(left == right);
}

void append_autotune_memory_sample(std::vector<autotune_memory_sample> &   samples,
                                   const std::string &                     checkpoint,
                                   const std::map<std::string, uint64_t> & free_memory_bytes) {
    autotune_memory_sample sample;
    sample.checkpoint        = checkpoint;
    sample.free_memory_bytes = free_memory_bytes;
    if (!samples.empty()) {
        const auto & baseline = samples.front().free_memory_bytes;
        const auto & previous = samples.back().free_memory_bytes;
        for (const auto & entry : free_memory_bytes) {
            const auto initial = baseline.find(entry.first);
            if (initial != baseline.end()) {
                sample.used_delta_bytes[entry.first] =
                    initial->second > entry.second ? initial->second - entry.second : 0;
            }
            const auto previous_free = previous.find(entry.first);
            if (previous_free != previous.end()) {
                sample.incremental_used_bytes[entry.first] =
                    previous_free->second > entry.second ? previous_free->second - entry.second : 0;
            }
        }
    } else {
        for (const auto & entry : free_memory_bytes) {
            sample.used_delta_bytes[entry.first]       = 0;
            sample.incremental_used_bytes[entry.first] = 0;
        }
    }
    samples.push_back(std::move(sample));
}

std::map<std::string, uint64_t> autotune_memory_peaks(const std::vector<autotune_memory_sample> & samples) {
    std::map<std::string, uint64_t> output;
    for (const auto & sample : samples) {
        for (const auto & entry : sample.used_delta_bytes) {
            output[entry.first] = std::max(output[entry.first], entry.second);
        }
    }
    return output;
}

std::map<std::string, uint64_t> autotune_measured_module_bytes(
    const std::vector<autotune_candidate_record> & candidates) {
    static const std::map<std::string, std::string> checkpoint_modules = {
        { "after_llm",       "llm"       },
        { "after_tts",       "tts"       },
        { "after_projector", "projector" },
        { "after_audio",     "audio"     },
        { "after_vision",    "vision"    },
        { "after_token2wav", "token2wav" },
    };
    std::map<std::string, uint64_t> output;
    auto                            consume = [&](const std::vector<autotune_memory_sample> & samples) {
        for (const auto & sample : samples) {
            const auto module = checkpoint_modules.find(sample.checkpoint);
            if (module == checkpoint_modules.end()) {
                continue;
            }
            uint64_t total = 0;
            for (const auto & device : sample.incremental_used_bytes) {
                total += device.second;
            }
            output[module->second] = std::max(output[module->second], total);
        }
    };
    for (const auto & candidate : candidates) {
        if (candidate.screening_performed) {
            consume(candidate.screening.memory_samples);
        }
        for (const auto & round : candidate.rounds) {
            consume(round.memory_samples);
        }
    }
    return output;
}

autotune_fingerprint create_autotune_fingerprint(const autotune_build_identity &            build,
                                                 const hardware_snapshot &                  hardware,
                                                 const std::vector<std::string> &           files,
                                                 const std::string &                        requested_profile,
                                                 const runtime_profile_overrides &          overrides,
                                                 std::string &                              error,
                                                 const std::map<std::string, std::string> & runtime_inputs,
                                                 const autotune_runtime_identity &          runtime_identity) {
    autotune_fingerprint output;
    output.build             = build;
    output.requested_profile = requested_profile;
    output.overrides         = overrides;
    output.runtime_inputs    = runtime_inputs;
    output.runtime_identity  = runtime_identity;
    error.clear();

    for (const auto & device : hardware.devices) {
        output.devices.push_back({
            device.name,
            device.description,
            device.is_accelerator,
            device.total_memory,
            device.type,
        });
    }
    std::sort(output.devices.begin(), output.devices.end(), [](const auto & left, const auto & right) {
        return std::tie(left.name, left.description, left.is_accelerator, left.total_memory, left.type) <
               std::tie(right.name, right.description, right.is_accelerator, right.total_memory, right.type);
    });

    for (const auto & file_name : files) {
        const std::filesystem::path path(file_name);
        std::error_code             filesystem_error;
        const auto                  absolute_path = std::filesystem::weakly_canonical(path, filesystem_error);
        if (filesystem_error) {
            error = "cannot canonicalize autotune input " + file_name + ": " + filesystem_error.message();
            return {};
        }
        const auto size = std::filesystem::file_size(absolute_path, filesystem_error);
        if (filesystem_error) {
            error = "cannot read autotune input size " + absolute_path.string() + ": " + filesystem_error.message();
            return {};
        }
        const auto modified = std::filesystem::last_write_time(absolute_path, filesystem_error);
        if (filesystem_error) {
            error =
                "cannot read autotune input timestamp " + absolute_path.string() + ": " + filesystem_error.message();
            return {};
        }
        const auto modified_time = static_cast<int64_t>(modified.time_since_epoch().count());
        output.files.push_back({ absolute_path.string(), size, modified_time });
    }
    std::sort(output.files.begin(), output.files.end(),
              [](const auto & left, const auto & right) { return left.path < right.path; });
    return output;
}

autotune_runtime_identity detect_autotune_runtime_identity(std::string & error) {
    autotune_runtime_identity output;
    error.clear();

    const auto executable = executable_path(error);
    if (!error.empty()) {
        return {};
    }
    std::error_code filesystem_error;
    const auto      canonical_executable = std::filesystem::weakly_canonical(executable, filesystem_error);
    if (filesystem_error) {
        error = "cannot canonicalize current executable " + executable + ": " + filesystem_error.message();
        return {};
    }
    if (!sha256_file(canonical_executable.string(), output.executable_sha256, error)) {
        return {};
    }

    std::map<std::string, autotune_backend_identity> libraries_by_path;
    const char *                                     proc_names[] = {
        "ggml_backend_get_features",
        "ggml_backend_set_n_threads",
        "ggml_backend_split_buffer_type",
    };
    for (size_t index = 0; index < ggml_backend_reg_count(); ++index) {
        const auto registry = ggml_backend_reg_get(index);
        void *     address  = nullptr;
        for (const char * proc_name : proc_names) {
            address = ggml_backend_reg_get_proc_address(registry, proc_name);
            if (address != nullptr) {
                break;
            }
        }
        const auto library_path = library_path_for_address(address);
        if (library_path.empty()) {
            continue;
        }
        const auto canonical_library = std::filesystem::weakly_canonical(library_path, filesystem_error);
        if (filesystem_error) {
            error = "cannot canonicalize GGML backend library " + library_path + ": " + filesystem_error.message();
            return {};
        }
        if (canonical_library == canonical_executable || libraries_by_path.count(canonical_library.string()) > 0) {
            continue;
        }
        autotune_backend_identity identity;
        if (!make_backend_identity(canonical_library.string(), identity, error)) {
            return {};
        }
        libraries_by_path.emplace(identity.path, std::move(identity));
    }
    for (auto & entry : libraries_by_path) {
        output.backend_libraries.push_back(std::move(entry.second));
    }

#ifdef _WIN32
    output.cuda_driver_version = query_cuda_version("cuDriverGetVersion", { "nvcuda.dll" });
    output.cuda_runtime_version =
        query_cuda_version("cudaRuntimeGetVersion", { "cudart64_130.dll", "cudart64_12.dll" });
#else
    output.cuda_driver_version = query_cuda_version("cuDriverGetVersion", { "libcuda.so.1", "libcuda.so" });
    output.cuda_runtime_version =
        query_cuda_version("cudaRuntimeGetVersion", { "libcudart.so", "libcudart.so.13", "libcudart.so.12" });
#endif
    return output;
}

bool write_autotune_cache(const std::string & path, const autotune_cache_record & record, std::string & error) {
    error.clear();
    const std::filesystem::path destination(path);
    if (destination.empty()) {
        error = "autotune cache path is empty";
        return false;
    }

    std::error_code filesystem_error;
    if (!destination.parent_path().empty()) {
        std::filesystem::create_directories(destination.parent_path(), filesystem_error);
        if (filesystem_error) {
            error = "cannot create autotune cache directory: " + filesystem_error.message();
            return false;
        }
    }

    static std::atomic<uint64_t> sequence{ 0 };
    auto                         temp_path = destination;
    temp_path += ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                 std::to_string(sequence.fetch_add(1));

    try {
        std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "cannot open autotune cache temporary file " + temp_path.string();
            return false;
        }
        output << serialize_cache(record).dump(2) << '\n';
        output.flush();
        if (!output) {
            error = "cannot write autotune cache temporary file " + temp_path.string();
            output.close();
            std::filesystem::remove(temp_path, filesystem_error);
            return false;
        }
        output.close();
    } catch (const std::exception & exception) {
        error = "cannot serialize autotune cache: " + std::string(exception.what());
        std::filesystem::remove(temp_path, filesystem_error);
        return false;
    }

    if (!replace_file(temp_path, destination, error)) {
        std::filesystem::remove(temp_path, filesystem_error);
        return false;
    }
    return true;
}

autotune_cache_read_result read_autotune_cache(const std::string &          path,
                                               const autotune_fingerprint & expected_fingerprint) {
    autotune_cache_read_result output;
    std::error_code            filesystem_error;
    if (!std::filesystem::exists(path, filesystem_error)) {
        output.status = autotune_cache_status::missing;
        output.reason = filesystem_error ? "cannot inspect autotune cache: " + filesystem_error.message() :
                                           "autotune cache does not exist";
        return output;
    }

    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            output.status = autotune_cache_status::invalid;
            output.reason = "cannot open autotune cache";
            return output;
        }
        const auto parsed = json::parse(input);
        output.record     = deserialize_cache(parsed);
    } catch (const std::exception & exception) {
        output.status = autotune_cache_status::invalid;
        output.reason = "cannot parse autotune cache: " + std::string(exception.what());
        return output;
    }

    if (output.record.schema_version != runtime_autotune_schema_version ||
        output.record.fingerprint.schema_version != runtime_autotune_schema_version) {
        output.status = autotune_cache_status::invalid;
        output.reason = "autotune cache schema version is unsupported";
        return output;
    }
    if (output.record.fingerprint != expected_fingerprint) {
        output.status = autotune_cache_status::stale;
        output.reason = "autotune cache fingerprint does not match the current runtime";
        return output;
    }
    const auto selected = std::find_if(
        output.record.candidates.begin(), output.record.candidates.end(),
        [&](const auto & candidate) { return candidate.id == output.record.selected_candidate && candidate.valid; });
    if (selected == output.record.candidates.end()) {
        output.status = autotune_cache_status::invalid;
        output.reason = "autotune cache does not contain a valid selected candidate";
        return output;
    }
    for (const auto & placement : selected->config.placements) {
        if (placement.status != "active" || placement.device == "cpu") {
            continue;
        }
        const auto device = std::find_if(expected_fingerprint.devices.begin(), expected_fingerprint.devices.end(),
                                         [&](const auto & available) { return available.name == placement.device; });
        if (device == expected_fingerprint.devices.end()) {
            output.status = autotune_cache_status::invalid;
            output.reason =
                "autotune cache selected unavailable device " + placement.device + " for module " + placement.module;
            return output;
        }
    }

    output.status = autotune_cache_status::applied;
    output.reason = "autotune cache fingerprint matched";
    return output;
}

autotune_candidate_plan create_autotune_candidate_plan(const effective_runtime_config &  base_config,
                                                       const hardware_snapshot &         hardware,
                                                       const runtime_profile_overrides & overrides,
                                                       const autotune_plan_options &     options) {
    autotune_candidate_plan output;
    const auto              accelerators = sorted_accelerators(hardware);
    if (accelerators.empty()) {
        output.skipped_reasons.push_back("runtime autotuning requires an accelerator");
        return output;
    }

    const runtime_device * primary = find_runtime_device(hardware, base_config.primary_device_name);
    if (primary == nullptr || !primary->is_accelerator) {
        primary = accelerators.front();
    }
    const runtime_device * secondary = nullptr;
    for (const auto * device : accelerators) {
        if (device->name != primary->name) {
            secondary = device;
            break;
        }
    }

    std::vector<autotune_candidate_record> candidates;
    if (!options.model_explicit && !overrides.llm_model.has_value()) {
        struct precision_input {
            const char *        id;
            const char *        precision;
            const std::string * model;
        };

        const precision_input precisions[] = {
            { "precision_f16",    "F16",    &options.inventory.llm_f16    },
            { "precision_q8_0",   "Q8_0",   &options.inventory.llm_q8_0   },
            { "precision_q4_k_m", "Q4_K_M", &options.inventory.llm_q4_k_m },
        };
        for (const auto & precision : precisions) {
            if (precision.model->empty()) {
                output.skipped_reasons.push_back(std::string(precision.id) +
                                                 " skipped because its GGUF model file is unavailable");
                continue;
            }
            if (base_config.llm_quantization == precision.precision) {
                continue;
            }
            autotune_candidate_record candidate;
            candidate.id                      = precision.id;
            candidate.stage                   = "precision";
            candidate.selectable              = options.allow_quantized;
            candidate.advisory                = !options.allow_quantized;
            candidate.config                  = base_config;
            candidate.config.resolved_profile = "autotune_" + candidate.id;
            candidate.config.reason           = "calibration candidate changes only the LLM GGUF precision";
            candidate.config.llm_model        = *precision.model;
            candidate.config.llm_quantization = precision.precision;
            for (auto & placement : candidate.config.placements) {
                if (placement.module == "llm") {
                    placement.model     = *precision.model;
                    placement.precision = precision.precision;
                }
            }
            candidates.push_back(std::move(candidate));
        }
    }

    auto add_placement = [&](const std::string & id, const std::string & encoder_device, const std::string & tts_device,
                             const std::string & projector_device, const std::string & token2wav_device,
                             int32_t token2wav_threads) {
        autotune_candidate_record candidate;
        candidate.id     = id;
        candidate.config = make_candidate_config(base_config, id, primary->name, encoder_device, tts_device,
                                                 projector_device, token2wav_device, overrides);
        if (!overrides.token2wav_threads.has_value()) {
            candidate.config.token2wav_threads = token2wav_threads;
        }
        candidates.push_back(std::move(candidate));
    };

    add_placement("consolidated", primary->name, primary->name, primary->name, primary->name,
                  base_config.token2wav_threads);

    if (secondary != nullptr) {
        add_placement("encoder_split", secondary->name, primary->name, secondary->name, primary->name,
                      base_config.token2wav_threads);
        add_placement("encoder_split_local_projector", secondary->name, primary->name, primary->name, primary->name,
                      base_config.token2wav_threads);
        add_placement("token2wav_split", secondary->name, primary->name, secondary->name, secondary->name,
                      base_config.token2wav_threads);
        add_placement("output_pipeline_split", secondary->name, secondary->name, secondary->name, secondary->name,
                      base_config.token2wav_threads);

        const uint32_t available_threads = options.hardware_concurrency > 0 ?
                                               options.hardware_concurrency :
                                               std::max(1U, std::thread::hardware_concurrency());
        for (const int32_t requested_threads : { 8, 16, 32, 64 }) {
            const int32_t threads = std::min<int32_t>(requested_threads, static_cast<int32_t>(available_threads));
            add_placement("gpu_cpu_hybrid_t" + std::to_string(requested_threads), secondary->name, primary->name,
                          primary->name, "cpu", threads);
        }
    } else if (secondary == nullptr) {
        output.skipped_reasons.push_back("split candidates require a secondary accelerator");
    }

    std::map<std::string, std::string> first_id_by_config;
    for (auto & candidate : candidates) {
        const auto key = candidate_config_key(candidate.config);
        if (first_id_by_config.count(key) > 0) {
            output.skipped_reasons.push_back(candidate.id + " duplicates " + first_id_by_config.at(key) +
                                             " after applying overrides and hardware limits");
            continue;
        }
        first_id_by_config.emplace(key, candidate.id);
        estimate_candidate_memory(candidate.config, options.previous_measured_module_bytes, candidate.memory_estimates);
        output.candidates.push_back(std::move(candidate));
    }

    for (const auto & candidate : output.candidates) {
        if (candidate.stage == "placement" && placement_matches(candidate.config, base_config) &&
            candidate.config.llm_model == base_config.llm_model &&
            candidate.config.token2wav_threads == base_config.token2wav_threads) {
            output.baseline_candidate = candidate.id;
            break;
        }
    }

    std::vector<autotune_candidate_record> filtered;
    for (auto & candidate : output.candidates) {
        std::string reason;
        const bool  baseline = candidate.id == output.baseline_candidate;
        if (!baseline && !candidate.memory_estimates.empty() && !candidate_fits_memory(candidate, hardware, reason)) {
            output.skipped_reasons.push_back(std::move(reason));
            continue;
        }
        filtered.push_back(std::move(candidate));
    }
    output.candidates = std::move(filtered);
    return output;
}

autotune_candidate_record evaluate_autotune_candidate(const std::string &                      id,
                                                      const effective_runtime_config &         config,
                                                      const autotune_calibration_observation & observation) {
    autotune_candidate_record output;
    output.id                  = id;
    output.config              = config;
    output.metrics             = observation.metrics;
    output.memory_samples      = observation.memory_samples;
    output.device_memory_peaks = autotune_memory_peaks(observation.memory_samples);
    output.response_sha256     = observation.response_sha256;
    output.tts_tokens_sha256   = observation.tts_tokens_sha256;
    output.tts_token_count     = observation.tts_token_count;
    output.modules_initialized = observation.modules_initialized;
    output.samples_finite      = observation.samples_finite;

    if (!observation.execution_error.empty()) {
        output.rejection_reason = observation.execution_error;
    } else if (!observation.modules_initialized) {
        output.rejection_reason = "one or more requested modules failed to initialize";
    } else if (observation.metrics.timeout_count > 0) {
        output.rejection_reason = "one or more calibration frames timed out";
    } else if (observation.metrics.frame_count < 12) {
        output.rejection_reason = "calibration did not complete all 12 fixture frames";
    } else if (observation.metrics.speak_count <= 0) {
        output.rejection_reason = "calibration did not produce a SPEAK transition";
    } else if (!observation.samples_finite || observation.metrics.audio_sample_count <= 0) {
        output.rejection_reason = "Token2Wav did not produce finite audio samples";
    } else if (observation.metrics.final_audio_count <= 0) {
        output.rejection_reason = "audio stream ended without is_final=true";
    } else if (!is_sha256_hex(observation.response_sha256) || !is_sha256_hex(observation.tts_tokens_sha256) ||
               observation.tts_token_count <= 0) {
        output.rejection_reason = "calibration did not produce valid response and TTS-token SHA-256 evidence";
    } else if (!metrics_are_finite(observation.metrics)) {
        output.rejection_reason = "calibration produced non-finite metrics";
    } else if (observation.metrics.frame_total_p95_ms >= 1000.0) {
        output.rejection_reason = "frame-total P95 must be below the 1000 ms fixture interval";
    } else if (observation.metrics.tts_rtf >= 1.0) {
        output.rejection_reason = "TTS RTF must be below 1.0";
    } else {
        output.score = policy_two_score(observation.metrics);
        output.valid = std::isfinite(output.score);
        if (!output.valid) {
            output.rejection_reason = "calibration score is not finite";
        }
    }
    return output;
}

autotune_candidate_record evaluate_autotune_candidate(
    const autotune_candidate_record &                     planned,
    const std::vector<autotune_calibration_observation> & observations) {
    autotune_candidate_record output = planned;
    output.metrics                   = {};
    output.rounds.clear();
    output.memory_samples.clear();
    output.device_memory_peaks.clear();
    output.modules_initialized = !observations.empty();
    output.samples_finite      = !observations.empty();
    output.valid               = !observations.empty();
    output.rejection_reason.clear();
    output.score                      = 0.0;
    output.output_consistency_checked = false;
    output.output_matches_baseline    = false;

    std::vector<double> frame_total;
    std::vector<double> encoder;
    std::vector<double> decode;
    std::vector<double> ttft;
    std::vector<double> first_audio;
    std::vector<double> rtf;
    for (size_t index = 0; index < observations.size(); ++index) {
        const auto &          observation = observations[index];
        const auto            evaluated   = evaluate_autotune_candidate(planned.id, planned.config, observation);
        autotune_round_record round;
        round.round_index         = static_cast<int32_t>(index);
        round.metrics             = observation.metrics;
        round.memory_samples      = observation.memory_samples;
        round.device_memory_peaks = autotune_memory_peaks(observation.memory_samples);
        round.response_sha256     = observation.response_sha256;
        round.tts_tokens_sha256   = observation.tts_tokens_sha256;
        round.tts_token_count     = observation.tts_token_count;
        round.modules_initialized = observation.modules_initialized;
        round.samples_finite      = observation.samples_finite;
        round.valid               = evaluated.valid;
        round.execution_error     = observation.execution_error;
        round.rejection_reason    = evaluated.rejection_reason;
        round.score               = evaluated.score;
        output.rounds.push_back(std::move(round));

        output.modules_initialized = output.modules_initialized && observation.modules_initialized;
        output.samples_finite      = output.samples_finite && observation.samples_finite;
        if (!evaluated.valid && output.valid) {
            output.valid            = false;
            output.rejection_reason = "round " + std::to_string(index) + ": " + evaluated.rejection_reason;
        }
        frame_total.push_back(observation.metrics.frame_total_p95_ms);
        encoder.push_back(observation.metrics.encoder_p95_ms);
        decode.push_back(observation.metrics.decode_p95_ms);
        ttft.push_back(observation.metrics.ttft_p95_ms);
        first_audio.push_back(observation.metrics.first_audio_p95_ms);
        rtf.push_back(observation.metrics.tts_rtf);
        for (const auto & sample : observation.memory_samples) {
            auto copied       = sample;
            copied.checkpoint = "round_" + std::to_string(index) + "/" + copied.checkpoint;
            output.memory_samples.push_back(std::move(copied));
        }
        for (const auto & peak : autotune_memory_peaks(observation.memory_samples)) {
            output.device_memory_peaks[peak.first] = std::max(output.device_memory_peaks[peak.first], peak.second);
        }
    }
    if (observations.empty()) {
        output.rejection_reason = "candidate has no measured rounds";
        return output;
    }

    output.metrics                    = observations.front().metrics;
    output.metrics.frame_total_p95_ms = median_plus_mad(frame_total);
    output.metrics.encoder_p95_ms     = median_plus_mad(encoder);
    output.metrics.decode_p95_ms      = median_plus_mad(decode);
    output.metrics.ttft_p95_ms        = median_plus_mad(ttft);
    output.metrics.first_audio_p95_ms = median_plus_mad(first_audio);
    output.metrics.tts_rtf            = median_plus_mad(rtf);
    output.metrics.timeout_count      = 0;
    for (const auto & observation : observations) {
        output.metrics.timeout_count += observation.metrics.timeout_count;
    }
    output.response_sha256   = observations.front().response_sha256;
    output.tts_tokens_sha256 = observations.front().tts_tokens_sha256;
    output.tts_token_count   = observations.front().tts_token_count;
    if (output.valid) {
        output.score = policy_two_score(output.metrics);
        if (!std::isfinite(output.score)) {
            output.valid            = false;
            output.rejection_reason = "aggregate calibration score is not finite";
        }
    }
    return output;
}

autotune_round_record evaluate_autotune_cpu_screening(const autotune_calibration_observation & observation) {
    autotune_round_record output;
    output.round_index         = -1;
    output.metrics             = observation.metrics;
    output.memory_samples      = observation.memory_samples;
    output.device_memory_peaks = autotune_memory_peaks(observation.memory_samples);
    output.response_sha256     = observation.response_sha256;
    output.tts_tokens_sha256   = observation.tts_tokens_sha256;
    output.tts_token_count     = observation.tts_token_count;
    output.modules_initialized = observation.modules_initialized;
    output.samples_finite      = observation.samples_finite;

    output.execution_error = observation.execution_error;
    if (!observation.execution_error.empty()) {
        output.rejection_reason = observation.execution_error;
    } else if (!observation.modules_initialized) {
        output.rejection_reason = "one or more requested modules failed to initialize during CPU screening";
    } else if (observation.metrics.timeout_count > 0) {
        output.rejection_reason = "one or more CPU screening frames timed out";
    } else if (observation.metrics.frame_count < 12) {
        output.rejection_reason = "CPU screening did not complete all 12 fixture frames";
    } else if (observation.metrics.speak_count <= 0) {
        output.rejection_reason = "CPU screening did not produce a SPEAK transition";
    } else if (!observation.samples_finite || observation.metrics.audio_sample_count <= 0) {
        output.rejection_reason = "Token2Wav did not produce finite audio samples during CPU screening";
    } else if (observation.metrics.final_audio_count <= 0) {
        output.rejection_reason = "CPU screening audio ended without is_final=true";
    } else if (!metrics_are_finite(observation.metrics)) {
        output.rejection_reason = "CPU screening produced non-finite metrics";
    } else if (observation.metrics.tts_rtf >= 1.5) {
        output.rejection_reason = "CPU screening TTS RTF must be below 1.5";
    } else {
        output.score = policy_two_score(observation.metrics);
        output.valid = std::isfinite(output.score);
        if (!output.valid) {
            output.rejection_reason = "CPU screening score is not finite";
        }
    }
    return output;
}

bool apply_autotune_output_consistency(std::vector<autotune_candidate_record> & candidates,
                                       const std::string &                      baseline_candidate) {
    const auto baseline = std::find_if(candidates.begin(), candidates.end(),
                                       [&](const auto & candidate) { return candidate.id == baseline_candidate; });
    if (baseline == candidates.end() || !baseline->valid || baseline->rounds.empty()) {
        return false;
    }
    for (auto & candidate : candidates) {
        if (!candidate.valid) {
            continue;
        }
        candidate.output_consistency_checked = true;
        bool matches                         = candidate.rounds.size() == baseline->rounds.size();
        for (size_t index = 0; matches && index < candidate.rounds.size(); ++index) {
            matches = candidate.rounds[index].response_sha256 == baseline->rounds[index].response_sha256 &&
                      candidate.rounds[index].tts_tokens_sha256 == baseline->rounds[index].tts_tokens_sha256;
        }
        candidate.output_matches_baseline = matches;
        if (!matches && candidate.stage != "precision") {
            candidate.valid = false;
            candidate.rejection_reason =
                "one or more paired rounds produced an output hash that differs from the Phase 2 baseline";
        }
    }
    return true;
}

std::vector<std::string> rotated_autotune_candidate_order(const std::vector<std::string> & candidate_ids,
                                                          int32_t                          round_index) {
    if (candidate_ids.empty()) {
        return {};
    }
    std::vector<std::string> output = candidate_ids;
    const size_t             offset = static_cast<size_t>(std::max(0, round_index)) % output.size();
    std::rotate(output.begin(), output.begin() + offset, output.end());
    return output;
}

autotune_schedule_result execute_autotune_schedule(const autotune_candidate_plan & plan,
                                                   int32_t                         measured_rounds,
                                                   int32_t                         warmup_rounds,
                                                   const autotune_run_callback &   run) {
    autotune_schedule_result output;
    if (measured_rounds <= 0 || warmup_rounds < 0 || !run) {
        return output;
    }

    std::map<std::string, std::vector<autotune_calibration_observation>> observations;
    std::map<std::string, autotune_round_record>                         screenings;
    std::map<std::string, bool>                                          active;
    for (const auto & candidate : plan.candidates) {
        active[candidate.id] = true;
        if (candidate.config.token2wav_device != "cpu") {
            continue;
        }
        std::string                      error;
        const autotune_schedule_request  request     = { true, -1, 0 };
        autotune_calibration_observation observation = run(candidate, request, error);
        observation.execution_error                  = error;
        auto screening                               = evaluate_autotune_cpu_screening(observation);
        screenings[candidate.id]                     = screening;
        if (!screening.valid) {
            active[candidate.id]                    = false;
            output.first_failed_round[candidate.id] = -1;
        }
    }

    std::vector<std::string> measured_candidate_ids;
    for (const auto & candidate : plan.candidates) {
        if (active[candidate.id]) {
            measured_candidate_ids.push_back(candidate.id);
        }
    }
    for (int32_t round_index = 0; round_index < measured_rounds; ++round_index) {
        const auto order = rotated_autotune_candidate_order(measured_candidate_ids, round_index);
        for (const auto & candidate_id : order) {
            if (!active[candidate_id]) {
                continue;
            }
            const auto                       candidate = std::find_if(plan.candidates.begin(), plan.candidates.end(),
                                                                      [&](const auto & value) { return value.id == candidate_id; });
            std::string                      error;
            const autotune_schedule_request  request     = { false, round_index, warmup_rounds };
            autotune_calibration_observation observation = run(*candidate, request, error);
            observation.execution_error                  = error;
            observations[candidate_id].push_back(observation);
            const auto evaluated = evaluate_autotune_candidate(candidate_id, candidate->config, observation);
            if (!evaluated.valid) {
                active[candidate_id]                    = false;
                output.first_failed_round[candidate_id] = round_index;
            }
        }
    }

    for (const auto & planned : plan.candidates) {
        auto       evaluated = evaluate_autotune_candidate(planned, observations[planned.id]);
        const auto screening = screenings.find(planned.id);
        if (screening != screenings.end()) {
            evaluated.screening_performed = true;
            evaluated.screening           = screening->second;
            evaluated.screening_rejected  = !screening->second.valid;
            if (evaluated.screening_rejected) {
                evaluated.valid            = false;
                evaluated.rejection_reason = "CPU screening rejected candidate: " + screening->second.rejection_reason;
            }
        }
        output.candidates.push_back(std::move(evaluated));
    }

    output.output_consistency_reference_available =
        apply_autotune_output_consistency(output.candidates, plan.baseline_candidate);
    const auto baseline = std::find_if(output.candidates.begin(), output.candidates.end(),
                                       [&](const auto & candidate) { return candidate.id == plan.baseline_candidate; });
    if (baseline != output.candidates.end()) {
        for (const auto & candidate : output.candidates) {
            if (candidate.output_matches_baseline || candidate.stage == "precision" ||
                output.first_failed_round.count(candidate.id) > 0) {
                continue;
            }
            const size_t paired_rounds = std::min(candidate.rounds.size(), baseline->rounds.size());
            for (size_t round_index = 0; round_index < paired_rounds; ++round_index) {
                if (candidate.rounds[round_index].response_sha256 != baseline->rounds[round_index].response_sha256 ||
                    candidate.rounds[round_index].tts_tokens_sha256 !=
                        baseline->rounds[round_index].tts_tokens_sha256) {
                    output.first_failed_round[candidate.id] = static_cast<int32_t>(round_index);
                    break;
                }
            }
        }
    }
    return output;
}

autotune_selection_result select_autotune_candidate(const std::vector<autotune_candidate_record> & candidates,
                                                    const std::string &                            baseline_candidate) {
    autotune_selection_result         output;
    const autotune_candidate_record * baseline = nullptr;
    for (const auto & candidate : candidates) {
        if (candidate.id == baseline_candidate) {
            baseline = &candidate;
        }
    }
    if (baseline == nullptr || !baseline->valid) {
        output.reason =
            "Phase 2 baseline is invalid; no valid baseline is available for paired scoring and output comparison";
        return output;
    }

    const autotune_candidate_record * best                            = baseline;
    bool                              lower_score_failed_paired_round = false;
    for (const auto & candidate : candidates) {
        if (&candidate == baseline || !candidate.valid || !candidate.selectable) {
            continue;
        }
        bool no_worse = true;
        if (!candidate.rounds.empty() || !baseline->rounds.empty()) {
            no_worse = candidate.rounds.size() == baseline->rounds.size();
            for (size_t index = 0; no_worse && index < candidate.rounds.size(); ++index) {
                no_worse = candidate.rounds[index].valid && baseline->rounds[index].valid &&
                           candidate.rounds[index].score <= baseline->rounds[index].score;
            }
        }
        if (!no_worse) {
            lower_score_failed_paired_round = lower_score_failed_paired_round || candidate.score < baseline->score;
            continue;
        }
        if (candidate.score < best->score) {
            best = &candidate;
        }
    }

    if (best != baseline && baseline->score > 0.0) {
        const double improvement = (baseline->score - best->score) / baseline->score;
        if (improvement < 0.05) {
            output.ok                 = true;
            output.selected_candidate = baseline->id;
            output.reason             = "best measured score improved by less than 5%; kept the Phase 2 baseline";
            return output;
        }
    }

    output.ok                 = true;
    output.selected_candidate = best->id;
    if (best != baseline) {
        output.reason = "selected the lowest valid score with at least 5% improvement and no worse paired round";
    } else if (lower_score_failed_paired_round) {
        output.reason =
            "kept the Phase 2 baseline because a lower aggregate score was worse in at least one paired round";
    } else {
        output.reason = "Phase 2 baseline had the lowest selectable valid score";
    }
    return output;
}

bool validate_autotune_cache_against_plan(const autotune_cache_record &   record,
                                          const autotune_candidate_plan & plan,
                                          std::string &                   error) {
    error.clear();
    if (record.candidates.size() != plan.candidates.size()) {
        error = "autotune cache candidate count does not match the current candidate plan";
        return false;
    }
    if (record.skipped_reasons != plan.skipped_reasons) {
        error = "autotune cache skipped-candidate reasons do not match the current candidate plan";
        return false;
    }

    int32_t    expected_rounds   = 0;
    const auto configured_rounds = record.fingerprint.runtime_inputs.find("autotune_rounds");
    if (configured_rounds != record.fingerprint.runtime_inputs.end()) {
        try {
            expected_rounds = std::stoi(configured_rounds->second);
        } catch (...) {
            error = "autotune cache contains an invalid autotune_rounds fingerprint value";
            return false;
        }
        if (expected_rounds < 1 || expected_rounds > 10) {
            error = "autotune cache contains an out-of-range autotune_rounds fingerprint value";
            return false;
        }
    }

    std::vector<autotune_candidate_record> recomputed_candidates;
    recomputed_candidates.reserve(plan.candidates.size());
    for (const auto & expected : plan.candidates) {
        const auto first = std::find_if(record.candidates.begin(), record.candidates.end(),
                                        [&](const auto & candidate) { return candidate.id == expected.id; });
        if (first == record.candidates.end() ||
            std::find_if(std::next(first), record.candidates.end(), [&](const auto & candidate) {
                return candidate.id == expected.id;
            }) != record.candidates.end()) {
            error = "autotune cache candidate IDs do not match the current candidate plan";
            return false;
        }
        if (!configs_match_for_cache(first->config, expected.config)) {
            error = "autotune cache candidate configuration does not match the current candidate plan";
            return false;
        }
        if (first->stage != expected.stage || first->selectable != expected.selectable ||
            first->advisory != expected.advisory || first->memory_estimates != expected.memory_estimates) {
            error = "autotune cache candidate policy metadata does not match the current candidate plan";
            return false;
        }

        const bool cpu_candidate = expected.config.token2wav_device == "cpu";
        if (first->screening_performed != cpu_candidate || (!first->screening_performed && first->screening_rejected)) {
            error = "autotune cache CPU screening metadata does not match the current candidate plan";
            return false;
        }

        autotune_candidate_record recomputed;
        autotune_round_record     recomputed_screening;
        if (first->screening_performed) {
            if (first->screening.round_index != -1) {
                error = "autotune cache CPU screening round index must be -1";
                return false;
            }
            recomputed_screening = evaluate_autotune_cpu_screening(observation_from_round(first->screening));
            if (!round_evaluation_matches(first->screening, recomputed_screening) ||
                first->screening_rejected == recomputed_screening.valid) {
                error = "autotune cache CPU screening result does not match its raw metrics";
                return false;
            }
        }
        if (!first->screening_rejected && first->rounds.empty()) {
            error = "autotune cache candidate has no raw measured round or CPU screening rejection";
            return false;
        }

        std::vector<autotune_calibration_observation> observations;
        observations.reserve(first->rounds.size());
        for (size_t round_index = 0; round_index < first->rounds.size(); ++round_index) {
            if (first->rounds[round_index].round_index != static_cast<int32_t>(round_index)) {
                error = "autotune cache measured round indexes are not contiguous";
                return false;
            }
            observations.push_back(observation_from_round(first->rounds[round_index]));
        }
        recomputed = evaluate_autotune_candidate(expected, observations);
        for (size_t round_index = 0; round_index < first->rounds.size(); ++round_index) {
            if (!round_evaluation_matches(first->rounds[round_index], recomputed.rounds[round_index])) {
                error = "autotune cache measured round result does not match its raw metrics";
                return false;
            }
        }

        if (first->screening_performed) {
            recomputed.screening_performed = true;
            recomputed.screening           = recomputed_screening;
            recomputed.screening_rejected  = !recomputed_screening.valid;
            if (recomputed.screening_rejected) {
                if (!first->rounds.empty()) {
                    error = "autotune cache CPU candidate contains measured rounds after screening rejection";
                    return false;
                }
                recomputed.valid = false;
                recomputed.rejection_reason =
                    "CPU screening rejected candidate: " + recomputed_screening.rejection_reason;
            }
        }

        if (expected_rounds > 0 && recomputed.valid && first->rounds.size() != static_cast<size_t>(expected_rounds)) {
            error = "autotune cache valid candidate did not complete every configured measured round";
            return false;
        }
        recomputed_candidates.push_back(std::move(recomputed));
    }

    if (!apply_autotune_output_consistency(recomputed_candidates, plan.baseline_candidate)) {
        error = "autotune cache Phase 2 baseline cannot establish paired output consistency";
        return false;
    }

    for (const auto & recomputed : recomputed_candidates) {
        const auto cached = std::find_if(record.candidates.begin(), record.candidates.end(),
                                         [&](const auto & candidate) { return candidate.id == recomputed.id; });
        if (cached->valid != recomputed.valid || cached->rejection_reason != recomputed.rejection_reason) {
            error = "autotune cache candidate validity does not match its raw measured rounds";
            return false;
        }
        if (!metrics_match(cached->metrics, recomputed.metrics) || !scalars_match(cached->score, recomputed.score)) {
            error = "autotune cache candidate score or aggregate metrics do not match its raw measured rounds";
            return false;
        }
        if (cached->modules_initialized != recomputed.modules_initialized ||
            cached->samples_finite != recomputed.samples_finite ||
            cached->response_sha256 != recomputed.response_sha256 ||
            cached->tts_tokens_sha256 != recomputed.tts_tokens_sha256 ||
            cached->tts_token_count != recomputed.tts_token_count ||
            serialize_memory_samples(cached->memory_samples) != serialize_memory_samples(recomputed.memory_samples) ||
            cached->device_memory_peaks != recomputed.device_memory_peaks) {
            error = "autotune cache candidate aggregate evidence does not match its raw measured rounds";
            return false;
        }
        if (cached->screening_performed != recomputed.screening_performed ||
            cached->screening_rejected != recomputed.screening_rejected ||
            cached->output_consistency_checked != recomputed.output_consistency_checked ||
            cached->output_matches_baseline != recomputed.output_matches_baseline) {
            error = "autotune cache candidate screening or output-consistency state is inconsistent";
            return false;
        }
    }

    const auto selection = select_autotune_candidate(recomputed_candidates, plan.baseline_candidate);
    if (!selection.ok || selection.selected_candidate != record.selected_candidate ||
        selection.reason != record.selection_reason) {
        error = "autotune cache selected candidate does not match the current scoring policy";
        return false;
    }
    return true;
}

}  // namespace omni
