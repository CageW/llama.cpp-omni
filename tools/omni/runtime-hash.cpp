#include "runtime-hash.h"

extern "C" {
#include "sha256/sha256.h"
}

#include <array>
#include <fstream>

namespace omni {
namespace {

std::string hex_digest(const unsigned char * digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string           output(SHA256_DIGEST_SIZE * 2, '0');
    for (size_t index = 0; index < SHA256_DIGEST_SIZE; ++index) {
        output[index * 2]     = digits[digest[index] >> 4];
        output[index * 2 + 1] = digits[digest[index] & 0x0f];
    }
    return output;
}

}  // namespace

std::string sha256_bytes(const void * data, size_t size) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, static_cast<const unsigned char *>(data), size);
    return hex_digest(digest);
}

bool sha256_file(const std::string & path, std::string & digest, std::string & error) {
    digest.clear();
    error.clear();
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open file for SHA-256: " + path;
        return false;
    }

    sha256_t context;
    sha256_init(&context);
    std::array<char, 1024 * 1024> buffer;
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            sha256_update(&context, reinterpret_cast<const unsigned char *>(buffer.data()), static_cast<size_t>(count));
        }
    }
    if (!input.eof()) {
        error = "cannot read file for SHA-256: " + path;
        return false;
    }

    unsigned char bytes[SHA256_DIGEST_SIZE];
    sha256_final(&context, bytes);
    digest = hex_digest(bytes);
    return true;
}

std::string normalized_response_sha256(const std::string & response) {
    std::string normalized;
    normalized.reserve(response.size());
    for (size_t index = 0; index < response.size(); ++index) {
        if (response[index] == '\r' && index + 1 < response.size() && response[index + 1] == '\n') {
            normalized.push_back('\n');
            ++index;
        } else {
            normalized.push_back(response[index]);
        }
    }
    auto is_ascii_whitespace = [](unsigned char value) {
        return value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f' || value == '\v';
    };
    size_t begin = 0;
    while (begin < normalized.size() && is_ascii_whitespace(static_cast<unsigned char>(normalized[begin]))) {
        ++begin;
    }
    size_t end = normalized.size();
    while (end > begin && is_ascii_whitespace(static_cast<unsigned char>(normalized[end - 1]))) {
        --end;
    }
    return sha256_bytes(normalized.data() + begin, end - begin);
}

std::string tts_token_sequence_sha256(const std::vector<int32_t> & tokens) {
    std::vector<unsigned char> bytes;
    bytes.reserve(tokens.size() * sizeof(int32_t));
    for (const int32_t token : tokens) {
        const uint32_t bits = static_cast<uint32_t>(token);
        bytes.push_back(static_cast<unsigned char>(bits & 0xff));
        bytes.push_back(static_cast<unsigned char>((bits >> 8) & 0xff));
        bytes.push_back(static_cast<unsigned char>((bits >> 16) & 0xff));
        bytes.push_back(static_cast<unsigned char>((bits >> 24) & 0xff));
    }
    return sha256_bytes(bytes.data(), bytes.size());
}

}  // namespace omni
