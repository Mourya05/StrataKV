#include "stratakv/tensor.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#ifdef _WIN32
#include <malloc.h>
#endif

namespace stratakv {
namespace {
void release_aligned(void* p) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
} // namespace

void Tensor::compute_strides() {
    const auto limit = static_cast<std::size_t>(
        (std::numeric_limits<std::ptrdiff_t>::max)()) / element_size();
    strides_.resize(shape_.size());
    std::size_t count = 1;
    for (std::size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = count;
        if (shape_[i] != 0 && count > limit / shape_[i])
            throw std::overflow_error("Tensor: size/stride overflow");
        count *= shape_[i];
    }
    total_elements_ = count;
}

Tensor::Tensor(const std::vector<std::size_t>& shape, DataType dtype)
    : dtype_(dtype), shape_(shape) {
    compute_strides();
    const std::size_t bytes = (std::max)(total_elements_, std::size_t{1}) *
        element_size();
    void* allocation = nullptr;
#ifdef _WIN32
    allocation = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&allocation, 64, bytes) != 0) allocation = nullptr;
#endif
    if (!allocation) throw std::bad_alloc();
    // shared_ptr invokes the deleter even if its control-block allocation fails.
    owner_ = std::shared_ptr<void>(allocation, release_aligned);
    data_ptr_ = allocation;
    std::memset(data_ptr_, 0, bytes);
}

Tensor::Tensor(const std::vector<std::size_t>& shape, float* data,
               std::shared_ptr<void> owner, AliasTag)
    : data_ptr_(data), dtype_(DataType::FP32), shape_(shape),
      owner_(std::move(owner)) {
    compute_strides();
    if (!owner_ || !data || reinterpret_cast<std::uintptr_t>(data) % 64 != 0)
        throw std::invalid_argument("Tensor: invalid owner or unaligned alias");
}

Tensor::Tensor(const std::vector<std::size_t>& shape, void* data,
               std::shared_ptr<void> owner, DataType dtype, AliasTag)
    : data_ptr_(data), dtype_(dtype), shape_(shape), owner_(std::move(owner)) {
    compute_strides();
    if (!owner_ || !data)
        throw std::invalid_argument("Tensor: invalid owner or null alias");
    if (dtype_ == DataType::FP32 &&
        reinterpret_cast<std::uintptr_t>(data) % 64 != 0)
        throw std::invalid_argument("Tensor: unaligned FP32 alias");
}

Tensor Tensor::from_external(const std::vector<std::size_t>& shape,
                             float* data, std::shared_ptr<void> owner) {
    return Tensor(shape, data, std::move(owner), AliasTag{});
}

Tensor Tensor::from_external_bytes(const std::vector<std::size_t>& shape,
                                   void* data, std::shared_ptr<void> owner,
                                   DataType dtype) {
    return Tensor(shape, data, std::move(owner), dtype, AliasTag{});
}

float Tensor::value_at(std::size_t i) const noexcept {
    if (dtype_ == DataType::FP32)
        return static_cast<const float*>(data_ptr_)[i];

    const auto* bytes = static_cast<const unsigned char*>(data_ptr_);
    const std::uint32_t bits =
        (static_cast<std::uint32_t>(bytes[2 * i]) |
         (static_cast<std::uint32_t>(bytes[2 * i + 1]) << 8u)) << 16u;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void Tensor::set_value(std::size_t i, float value) noexcept {
    if (dtype_ == DataType::FP32) {
        static_cast<float*>(data_ptr_)[i] = value;
        return;
    }

    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    std::uint32_t upper = bits >> 16u;
    const std::uint32_t lower = bits & 0xffffu;
    // Round to nearest, ties to even, which is the IEEE-compatible BF16
    // conversion used by the CUDA intrinsics.
    if (lower > 0x8000u || (lower == 0x8000u && (upper & 1u) != 0u))
        ++upper;
    auto* bytes = static_cast<unsigned char*>(data_ptr_);
    bytes[2 * i] = static_cast<unsigned char>(upper & 0xffu);
    bytes[2 * i + 1] = static_cast<unsigned char>(upper >> 8u);
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_ptr_(std::exchange(other.data_ptr_, nullptr)),
      dtype_(other.dtype_),
      total_elements_(std::exchange(other.total_elements_, 0)),
      shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      owner_(std::move(other.owner_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        owner_ = std::move(other.owner_);
        data_ptr_ = std::exchange(other.data_ptr_, nullptr);
        dtype_ = other.dtype_;
        total_elements_ = std::exchange(other.total_elements_, 0);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
    }
    return *this;
}

std::size_t Tensor::get_flat_index(const std::vector<std::size_t>& idx) const {
    if (idx.size() != shape_.size())
        throw std::invalid_argument("Tensor: index rank mismatch");
    std::size_t flat = 0;
    for (std::size_t i = 0; i < idx.size(); ++i) {
#ifndef NDEBUG
        if (idx[i] >= shape_[i])
            throw std::out_of_range("Tensor: index at dimension " + std::to_string(i));
#endif
        flat += idx[i] * strides_[i];
    }
    return flat;
}
} // namespace stratakv
