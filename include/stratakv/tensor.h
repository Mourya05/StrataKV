#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

namespace stratakv {

// ============================================================
//  Tensor  --  contiguous, 64-byte-aligned, row-major storage
// ============================================================
class Tensor {
public:
    // ---- Construction / Destruction ----------------------------------------

    /// Allocate a tensor with the given shape.
    /// Memory is 64-byte aligned so AVX2/AVX-512 loads/stores are safe.
    explicit Tensor(const std::vector<size_t>& shape);

    /// Rule-of-Five: destructor frees aligned allocation.
    ~Tensor();

    // Copy is disabled -- tensors can be large; force explicit handling.
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move semantics: O(1), steals the allocation.
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // ---- Accessors ---------------------------------------------------------

    /// Raw pointer to the first element (64-byte aligned).
    float*       data()       noexcept { return data_ptr_; }
    const float* data() const noexcept { return data_ptr_; }

    /// Total number of float elements across all dimensions.
    size_t num_elements() const noexcept { return total_elements_; }

    /// Shape of the tensor (e.g., {M, N} for a 2-D matrix).
    const std::vector<size_t>& shape()   const noexcept { return shape_; }

    /// Row-major strides in number-of-floats (not bytes).
    const std::vector<size_t>& strides() const noexcept { return strides_; }

    /// Number of dimensions.
    size_t ndim() const noexcept { return shape_.size(); }

    /// Compute the flat 1-D index for an N-D index vector.
    /// Performs bounds checking in debug builds (NDEBUG strips it).
    size_t get_flat_index(const std::vector<size_t>& idx) const;

    /// Convenience: element access by flat index (no bounds check).
    float& operator[](size_t flat_idx)       noexcept { return data_ptr_[flat_idx]; }
    float  operator[](size_t flat_idx) const noexcept { return data_ptr_[flat_idx]; }

private:
    float*              data_ptr_       = nullptr;
    size_t              total_elements_ = 0;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    /// Populate strides_ from shape_ using row-major convention.
    void compute_strides();
};

} // namespace stratakv
