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
#ifdef STRATAKV_HAS_CUDA
#include <cuda_runtime_api.h>
#endif
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

#ifdef STRATAKV_HAS_CUDA
void release_cuda(void* p) noexcept {
    if (p) (void)cudaFree(p);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor: ") + operation +
                                 " failed: " + cudaGetErrorString(status));
    }
}
#endif
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

Tensor::Tensor(const std::vector<std::size_t>& shape, DataType dtype,
               Device device)
    : device_(device), dtype_(dtype), shape_(shape) {
    compute_strides();
    const std::size_t bytes = (std::max)(total_elements_, std::size_t{1}) *
        element_size();
    void* allocation = nullptr;

    if (device_ == Device::GPU) {
#ifdef STRATAKV_HAS_CUDA
        cuda_check(cudaMalloc(&allocation, bytes), "cudaMalloc");
        owner_ = std::shared_ptr<void>(allocation, release_cuda);
        data_ptr_ = allocation;
        if (total_elements_ != 0)
            cuda_check(cudaMemset(data_ptr_, 0, bytes), "cudaMemset");
        return;
#else
        throw std::runtime_error(
            "Tensor: GPU storage requested but CUDA support is disabled");
#endif
    }

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
    : data_ptr_(data), device_(Device::CPU), dtype_(DataType::FP32), shape_(shape),
      owner_(std::move(owner)) {
    compute_strides();
    if (!owner_ || !data || reinterpret_cast<std::uintptr_t>(data) % 64 != 0)
        throw std::invalid_argument("Tensor: invalid owner or unaligned alias");
}

Tensor::Tensor(const std::vector<std::size_t>& shape, void* data,
               std::shared_ptr<void> owner, DataType dtype, AliasTag)
    : data_ptr_(data), device_(Device::CPU), dtype_(dtype), shape_(shape),
      owner_(std::move(owner)) {
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

Tensor::Tensor(Tensor&& other) noexcept
    : data_ptr_(std::exchange(other.data_ptr_, nullptr)),
      device_(other.device_),
      dtype_(other.dtype_),
      total_elements_(std::exchange(other.total_elements_, 0)),
      shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      owner_(std::move(other.owner_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        owner_ = std::move(other.owner_);
        data_ptr_ = std::exchange(other.data_ptr_, nullptr);
        device_ = other.device_;
        dtype_ = other.dtype_;
        total_elements_ = std::exchange(other.total_elements_, 0);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
    }
    return *this;
}

Tensor::Tensor(const Tensor& source, AliasTag)
    : data_ptr_(source.data_ptr_), device_(source.device_), dtype_(source.dtype_),
      total_elements_(source.total_elements_), shape_(source.shape_),
      strides_(source.strides_), owner_(source.owner_) {}

Tensor Tensor::transpose_2d() const {
    if (ndim() != 2 || dtype_ != DataType::BF16 || !owner_)
        throw std::invalid_argument("Tensor::transpose_2d: expected a BF16 matrix");
    Tensor view(*this, AliasTag{});
    std::swap(view.shape_[0], view.shape_[1]);
    std::swap(view.strides_[0], view.strides_[1]);
    return view;
}

std::size_t Tensor::storage_index(std::size_t i) const noexcept {
    if (is_contiguous()) return i;
    return (i / shape_[1]) * strides_[0] + (i % shape_[1]) * strides_[1];
}

Tensor Tensor::to(Device target) const {
    Tensor result(shape_, dtype_, target);
    result.strides_ = strides_; // Copy the physical buffer and its layout.
    const std::size_t bytes = total_elements_ * element_size();
    if (bytes == 0) return result;

    if (device_ == Device::CPU && target == Device::CPU) {
        std::memcpy(result.raw_data(), data_ptr_, bytes);
        return result;
    }

#ifdef STRATAKV_HAS_CUDA
    cudaMemcpyKind kind;
    if (device_ == Device::CPU && target == Device::GPU)
        kind = cudaMemcpyHostToDevice;
    else if (device_ == Device::GPU && target == Device::CPU)
        kind = cudaMemcpyDeviceToHost;
    else if (device_ == Device::GPU && target == Device::GPU)
        kind = cudaMemcpyDeviceToDevice;
    else
        throw std::logic_error("Tensor: unsupported device transfer");
    cuda_check(cudaMemcpy(result.raw_data(), data_ptr_, bytes, kind),
               "cudaMemcpy");
    return result;
#else
    (void)target;
    throw std::runtime_error(
        "Tensor: GPU transfer requested but CUDA support is disabled");
#endif
}

float Tensor::value_at(std::size_t i) const {
    if (device_ != Device::CPU)
        throw std::runtime_error("Tensor: GPU tensor requires to(Device::CPU)");
    i = storage_index(i);
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

void Tensor::set_value(std::size_t i, float value) {
    if (device_ != Device::CPU)
        throw std::runtime_error("Tensor: GPU tensor requires to(Device::CPU)");
    i = storage_index(i);
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

std::size_t Tensor::get_flat_index(const std::vector<std::size_t>& idx) const {
    if (idx.size() != shape_.size())
        throw std::invalid_argument("Tensor: index rank mismatch");
    std::size_t flat = 0;
    for (std::size_t i = 0; i < idx.size(); ++i) {
#ifndef NDEBUG
        if (idx[i] >= shape_[i])
            throw std::out_of_range("Tensor: index at dimension " + std::to_string(i));
#endif
        // operator[] consumes a logical row-major flat index, including for
        // transposed BF16 views; storage_index translates it to physical bytes.
        flat = flat * shape_[i] + idx[i];
    }
    return flat;
}
} // namespace stratakv
