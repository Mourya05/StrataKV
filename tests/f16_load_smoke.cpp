#include "stratakv/io/safetensors.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void write_file(const std::filesystem::path& path,
                const std::string& header,
                const std::vector<unsigned char>& payload) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (unsigned i = 0; i < 8; ++i)
        output.put(static_cast<char>((size >> (8u * i)) & 0xffu));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("supply a scratch directory");
        constexpr std::size_t count = 1024u * 1024u;
        std::vector<unsigned char> payload(count * 2u);
        for (std::size_t i = 0; i < count; ++i) {
            payload[2u * i] = 0x00;
            payload[2u * i + 1u] = (i & 1u) != 0u ? 0xc1 : 0x3c;
        }
        const auto file = std::filesystem::path(argv[1]) /
            "f16_parallel_load.safetensors";
        write_file(file,
            R"({"large":{"dtype":"F16","shape":[1048576],"data_offsets":[0,2097152]}})",
            payload);
        const auto weights = stratakv::io::load_safetensors(file);
        const auto& tensor = weights.at("large");
        if (tensor.dtype() != stratakv::DataType::FP32 ||
            tensor[0] != 1.0f || tensor[1] != -2.5f ||
            tensor[count - 1] != -2.5f) {
            throw std::runtime_error("parallel F16 decode produced wrong values");
        }
        const auto f16_timing = stratakv::io::last_load_breakdown();

        std::vector<unsigned char> bf16_payload(count * 2u);
        for (std::size_t i = 0; i < count; ++i) {
            bf16_payload[2u * i] = (i & 1u) != 0u ? 0x20 : 0x80;
            bf16_payload[2u * i + 1u] = (i & 1u) != 0u ? 0xc0 : 0x3f;
        }
        write_file(file,
            R"({"large":{"dtype":"BF16","shape":[1048576],"data_offsets":[0,2097152]}})",
            bf16_payload);
        const auto bf16_weights = stratakv::io::load_safetensors(file);
        const auto& bf16_tensor = bf16_weights.at("large");
        if (bf16_tensor.dtype() != stratakv::DataType::FP32 ||
            bf16_tensor[0] != 1.0f || bf16_tensor[1] != -2.5f ||
            bf16_tensor[count - 1] != -2.5f) {
            throw std::runtime_error("parallel BF16 decode produced wrong values");
        }
        const auto bf16_timing = stratakv::io::last_load_breakdown();
        std::cout << "F16/BF16 parallel load: passed, f16_tensor_decode_ms = "
                  << f16_timing.tensor_decode_ms
                  << ", bf16_tensor_decode_ms = "
                  << bf16_timing.tensor_decode_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "F16 parallel load failure: " << error.what() << '\n';
        return 1;
    }
}
