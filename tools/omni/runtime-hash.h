#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace omni {

std::string sha256_bytes(const void * data, size_t size);
bool        sha256_file(const std::string & path, std::string & digest, std::string & error);
std::string normalized_response_sha256(const std::string & response);
std::string tts_token_sequence_sha256(const std::vector<int32_t> & tokens);

}  // namespace omni
