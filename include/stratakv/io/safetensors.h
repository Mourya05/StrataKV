#pragma once
#include "stratakv/tensor.h"
#include <filesystem>
#include <string>
#include <unordered_map>

namespace stratakv::io {
using WeightMap = std::unordered_map<std::string, Tensor>;

struct LoadBreakdown {
    double mmap_open_ms = 0.0;
    double header_parse_ms = 0.0;
    double tensor_decode_ms = 0.0;
};

// All tensors returned through the compatibility API are materialized as
// FP32, while aligned F32 norm tensors may be safely aliased.
// Aligned FP32 norm tensors may be aliased without copying.
// true rejects non-mappable FP32 tensors.
// All returned aliases retain the mapping through Tensor's storage owner.
WeightMap load_safetensors(const std::filesystem::path& path,
                          bool require_mapped_f32 = false);
LoadBreakdown last_load_breakdown() noexcept;

// Convert a checkpoint linear matrix from [out_features, in_features] to the
// GEMM layout [in_features, out_features] directly into BF16 storage.
Tensor load_linear_bf16_transposed(const Tensor& source,
                                   std::size_t out_features,
                                   std::size_t in_features);
} // namespace stratakv::io
