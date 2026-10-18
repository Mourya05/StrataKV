#pragma once
#include "stratakv/tensor.h"
#include <filesystem>
#include <string>
#include <unordered_map>

namespace stratakv::io {
using WeightMap = std::unordered_map<std::string, Tensor>;

// Norm/gamma weights are retained as FP32. Other weights are normalized to
// BF16 storage, including F16 values decoded from IEEE binary16 first.
// Aligned FP32 norm tensors may be aliased without copying.
// true rejects non-mappable FP32 tensors.
// All returned aliases retain the mapping through Tensor's storage owner.
WeightMap load_safetensors(const std::filesystem::path& path,
                          bool require_mapped_f32 = false);
} // namespace stratakv::io
