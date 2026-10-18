#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace stratakv {

enum class DataType {
    FP32,
    BF16,
};

class Tensor {
public:
    explicit Tensor(const std::vector<std::size_t>& shape,
                    DataType dtype = DataType::FP32);
    ~Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Lifetime and alignment are checked. The owner must keep the entire
    // referenced range alive; data may be private copy-on-write storage.
    static Tensor from_external(const std::vector<std::size_t>& shape,
                                float* data, std::shared_ptr<void> owner);
    static Tensor from_external_bytes(const std::vector<std::size_t>& shape,
                                      void* data, std::shared_ptr<void> owner,
                                      DataType dtype);

    DataType dtype() const noexcept { return dtype_; }
    std::size_t element_size() const noexcept {
        return dtype_ == DataType::BF16 ? sizeof(std::uint16_t) : sizeof(float);
    }

    // data() is intentionally limited to FP32 tensors so an accidental raw
    // BF16-as-float access cannot silently corrupt memory. Use raw_data() for
    // byte-wise copies and operator[]/value_at() for converted scalar access.
    float* data() noexcept {
        return dtype_ == DataType::FP32 ? static_cast<float*>(data_ptr_) : nullptr;
    }
    const float* data() const noexcept {
        return dtype_ == DataType::FP32 ? static_cast<const float*>(data_ptr_) : nullptr;
    }
    void* raw_data() noexcept { return data_ptr_; }
    const void* raw_data() const noexcept { return data_ptr_; }
    std::size_t num_elements() const noexcept { return total_elements_; }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    std::size_t ndim() const noexcept { return shape_.size(); }
    std::size_t get_flat_index(const std::vector<std::size_t>& idx) const;

    class ElementProxy {
    public:
        ElementProxy(Tensor& tensor, std::size_t index) noexcept
            : tensor_(&tensor), index_(index) {}
        operator float() const noexcept { return tensor_->value_at(index_); }
        ElementProxy& operator=(float value) noexcept {
            tensor_->set_value(index_, value);
            return *this;
        }
        ElementProxy& operator=(const ElementProxy& other) noexcept {
            return *this = static_cast<float>(other);
        }

    private:
        Tensor* tensor_;
        std::size_t index_;
    };

    ElementProxy operator[](std::size_t i) noexcept {
        return ElementProxy(*this, i);
    }
    float operator[](std::size_t i) const noexcept { return value_at(i); }
    float value_at(std::size_t i) const noexcept;

private:
    struct AliasTag {};
    Tensor(const std::vector<std::size_t>& shape, float* data,
           std::shared_ptr<void> owner, AliasTag);
    Tensor(const std::vector<std::size_t>& shape, void* data,
           std::shared_ptr<void> owner, DataType dtype, AliasTag);
    void compute_strides();
    void set_value(std::size_t i, float value) noexcept;
    void* data_ptr_ = nullptr;
    DataType dtype_ = DataType::FP32;
    std::size_t total_elements_ = 0;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::shared_ptr<void> owner_;
};
} // namespace stratakv
