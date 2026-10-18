#pragma once
#include "stratakv/tensor.h"
#include <filesystem>
#include <string>
#include <unordered_map>

namespace stratakv::io {
using WeightMap = std::unordered_map<std::string, Tensor>;

// F32 is aliased if little-endian and 64-byte aligned, otherwise copied.
// BF16/bfloat16 is retained as native two-byte storage and is aliased when
// possible, so loading it does not expand the checkpoint in host memory.
// true rejects non-mappable F32 tensors.
// All returned aliases retain the mapping through Tensor's storage owner.
WeightMap load_safetensors(const std::filesystem::path& path,
                          bool require_mapped_f32 = false);
} // namespace stratakv::io
