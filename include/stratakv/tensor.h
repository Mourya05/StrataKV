#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace stratakv {
class Tensor {
public:
    explicit Tensor(const std::vector<std::size_t>& shape);
    ~Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Lifetime and alignment are checked. The owner must keep the entire
    // referenced float range alive; data may be private copy-on-write storage.
    static Tensor from_external(const std::vector<std::size_t>& shape,
                                float* data, std::shared_ptr<void> owner);
    float* data() noexcept { return data_ptr_; }
    const float* data() const noexcept { return data_ptr_; }
    std::size_t num_elements() const noexcept { return total_elements_; }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    std::size_t ndim() const noexcept { return shape_.size(); }
    std::size_t get_flat_index(const std::vector<std::size_t>& idx) const;
    float& operator[](std::size_t i) noexcept { return data_ptr_[i]; }
    float operator[](std::size_t i) const noexcept { return data_ptr_[i]; }

private:
    struct AliasTag {};
    Tensor(const std::vector<std::size_t>& shape, float* data,
           std::shared_ptr<void> owner, AliasTag);
    void compute_strides();
    float* data_ptr_ = nullptr;
    std::size_t total_elements_ = 0;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::shared_ptr<void> owner_;
};
} // namespace stratakv