#include "stratakv/generation/sampler.h"
#include "stratakv/io/safetensors.h"
#include "stratakv/ops.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cmath>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void write_bf16_file(const std::filesystem::path& path) {
    const std::string header =
        R"({"x":{"dtype":"bfloat16","shape":[3],"data_offsets":[0,6]}})";
    const std::vector<unsigned char> bytes = {0x00, 0x3f, 0x80, 0xbf,
                                              0x00, 0x40};
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (unsigned i = 0; i < 8; ++i)
        output.put(static_cast<char>((size >> (8u * i)) & 0xffu));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("supply a scratch directory");

        stratakv::Tensor values({3}, stratakv::DataType::BF16);
        values[0] = 1.0f;
        values[1] = -2.5f;
        values[2] = 3.25f;
        check(values.dtype() == stratakv::DataType::BF16,
              "BF16 tensor dtype was not retained");
        check(values.data() == nullptr, "BF16 exposed an FP32 pointer");
        const auto* raw = static_cast<const unsigned char*>(values.raw_data());
        check(raw[0] == 0x80 && raw[1] == 0x3f &&
              raw[2] == 0x20 && raw[3] == 0xc0,
              "BF16 storage bytes were not preserved");
        check(values[0] == 1.0f && values[1] == -2.5f && values[2] == 3.25f,
              "BF16 scalar conversion failed");

        const auto file = std::filesystem::path(argv[1]) / "phase7_bf16.safetensors";
        write_bf16_file(file);
        auto weights = stratakv::io::load_safetensors(file);
        check(weights.at("x").dtype() == stratakv::DataType::BF16,
              "safetensors expanded bfloat16 instead of mapping it");
        check(weights.at("x")[0] == 0.5f && weights.at("x")[1] == -1.0f &&
              weights.at("x")[2] == 2.0f,
              "mapped bfloat16 values were decoded incorrectly");

        stratakv::Tensor logits({4});
        logits[0] = 0.0f;
        logits[1] = 1.0f;
        logits[2] = 2.0f;
        logits[3] = 3.0f;
        stratakv::generation::Sampler sampler(
            stratakv::generation::SamplerConfig{1.0f, 2, 1.0f}, 7u);
        for (int i = 0; i < 32; ++i) {
            const int token = sampler.sample(logits);
            check(token == 2 || token == 3, "Top-K emitted a filtered token");
        }

        stratakv::generation::Sampler nucleus(
            stratakv::generation::SamplerConfig{1.0f, 0, 0.5f}, 7u);
        for (int i = 0; i < 8; ++i)
            check(nucleus.sample(logits) == 3,
                  "Top-P did not retain the highest-probability token");

#if defined(STRATAKV_HAS_CUDA)
        stratakv::Tensor left({2, 3}, stratakv::DataType::BF16);
        stratakv::Tensor right({3, 2}, stratakv::DataType::BF16);
        const float left_values[] = {1, 2, 3, 4, 5, 6};
        const float right_values[] = {7, 8, 9, 10, 11, 12};
        for (std::size_t i = 0; i < 6; ++i) left[i] = left_values[i];
        for (std::size_t i = 0; i < 6; ++i) right[i] = right_values[i];
        stratakv::Tensor product({2, 2});
        stratakv::matmul_cuda_tiled(left, right, product);
        check(std::fabs(product[0] - 58.0f) < 1.0e-3f &&
              std::fabs(product[1] - 64.0f) < 1.0e-3f &&
              std::fabs(product[2] - 139.0f) < 1.0e-3f &&
              std::fabs(product[3] - 154.0f) < 1.0e-3f,
              "CUDA BF16 GEMM result mismatch");
#endif

        std::cout << "Phase 7: BF16 and sampler smoke checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Phase 7 failure: " << error.what() << '\n';
        return 1;
    }
}
