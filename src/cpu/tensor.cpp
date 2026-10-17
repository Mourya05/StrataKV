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
        (std::numeric_limits<std::ptrdiff_t>::max)()) / sizeof(float);
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

Tensor::Tensor(const std::vector<std::size_t>& shape) : shape_(shape) {
    compute_strides();
    const std::size_t bytes = (std::max)(total_elements_, std::size_t{1}) * sizeof(float);
    void* allocation = nullptr;
#ifdef _WIN32
    allocation = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&allocation, 64, bytes) != 0) allocation = nullptr;
#endif
    if (!allocation) throw std::bad_alloc();
    // shared_ptr invokes the deleter even if its control-block allocation fails.
    owner_ = std::shared_ptr<void>(allocation, release_aligned);
    data_ptr_ = static_cast<float*>(allocation);
    std::memset(data_ptr_, 0, bytes);
}

Tensor::Tensor(const std::vector<std::size_t>& shape, float* data,
               std::shared_ptr<void> owner, AliasTag)
    : data_ptr_(data), shape_(shape), owner_(std::move(owner)) {
    compute_strides();
    if (!owner_ || !data || reinterpret_cast<std::uintptr_t>(data) % 64 != 0)
        throw std::invalid_argument("Tensor: invalid owner or unaligned alias");
}

Tensor Tensor::from_external(const std::vector<std::size_t>& shape,
                             float* data, std::shared_ptr<void> owner) {
    return Tensor(shape, data, std::move(owner), AliasTag{});
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_ptr_(std::exchange(other.data_ptr_, nullptr)),
      total_elements_(std::exchange(other.total_elements_, 0)),
      shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      owner_(std::move(other.owner_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        owner_ = std::move(other.owner_);
        data_ptr_ = std::exchange(other.data_ptr_, nullptr);
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