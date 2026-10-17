#include "stratakv/tensor.h"

#include <malloc.h>      // _aligned_malloc / _aligned_free (MSVC/Windows)
#include <stdexcept>
#include <numeric>       // std::accumulate
#include <cstring>       // std::memset

namespace stratakv {

// ---- Helpers ---------------------------------------------------------------

static constexpr size_t kAlignment = 64; // cache-line + AVX-512 width

// ---- Constructor -----------------------------------------------------------

Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape)
{
    if (shape_.empty()) {
        throw std::invalid_argument("Tensor shape must have at least one dimension.");
    }

    // Compute total element count (product of all dims).
    total_elements_ = 1;
    for (size_t dim : shape_) {
        if (dim == 0) {
            throw std::invalid_argument("Tensor shape dimensions must be non-zero.");
        }
        total_elements_ *= dim;
    }

    // 64-byte aligned allocation -- mandatory on Windows/MSVC.
    const size_t bytes = total_elements_ * sizeof(float);
    data_ptr_ = static_cast<float*>(_aligned_malloc(bytes, kAlignment));
    if (data_ptr_ == nullptr) {
        throw std::bad_alloc();
    }

    // Zero-initialise so uninitialised reads surface as 0.0f rather than NaN.
    std::memset(data_ptr_, 0, bytes);

    compute_strides();
}

// ---- Destructor ------------------------------------------------------------

Tensor::~Tensor() {
    if (data_ptr_) {
        _aligned_free(data_ptr_);
        data_ptr_ = nullptr;
    }
}

// ---- Move constructor ------------------------------------------------------

Tensor::Tensor(Tensor&& other) noexcept
    : data_ptr_      (other.data_ptr_)
    , total_elements_(other.total_elements_)
    , shape_         (std::move(other.shape_))
    , strides_       (std::move(other.strides_))
{
    // Leave the moved-from object in a valid, destructible state.
    other.data_ptr_       = nullptr;
    other.total_elements_ = 0;
}

// ---- Move assignment -------------------------------------------------------

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Release current allocation before stealing the other's.
        if (data_ptr_) {
            _aligned_free(data_ptr_);
        }
        data_ptr_       = other.data_ptr_;
        total_elements_ = other.total_elements_;
        shape_          = std::move(other.shape_);
        strides_        = std::move(other.strides_);

        other.data_ptr_       = nullptr;
        other.total_elements_ = 0;
    }
    return *this;
}

// ---- compute_strides -------------------------------------------------------
//
//  Row-major (C-contiguous) strides:
//
//    For shape {d0, d1, ..., dN-1}:
//      stride[N-1] = 1
//      stride[i]   = stride[i+1] * shape[i+1]
//
//  Example -- shape {2, 3, 4}:
//    stride[2] = 1
//    stride[1] = 1 * 4 = 4
//    stride[0] = 4 * 3 = 12
//  -> strides {12, 4, 1}

void Tensor::compute_strides() {
    const size_t ndims = shape_.size();
    strides_.resize(ndims);

    if (ndims == 0) return;

    strides_[ndims - 1] = 1;
    for (size_t i = ndims - 1; i-- > 0; ) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// ---- get_flat_index --------------------------------------------------------

size_t Tensor::get_flat_index(const std::vector<size_t>& idx) const {
    if (idx.size() != shape_.size()) {
        throw std::invalid_argument(
            "Index dimensionality does not match tensor rank.");
    }

    size_t flat = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
#ifndef NDEBUG
        if (idx[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of range at dimension " +
                                    std::to_string(i));
        }
#endif
        flat += idx[i] * strides_[i];
    }
    return flat;
}

} // namespace stratakv
