#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace stratakv {

enum class Device {
    CPU,
    GPU,
};

enum class DataType {
    FP32,
    BF16,
};

class Tensor {
public:
    explicit Tensor(const std::vector<std::size_t>& shape,
                    DataType dtype = DataType::FP32,
                    Device device = Device::CPU);
    explicit Tensor(const std::vector<std::size_t>& shape, Device device)
        : Tensor(shape, DataType::FP32, device) {}
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

    Device device() const noexcept { return device_; }
    // Return an owning deep copy on the requested device. The source tensor
    // remains unchanged, which makes failed transfers transactional.
    Tensor to(Device target) const;
    DataType dtype() const noexcept { return dtype_; }
    std::size_t element_size() const noexcept {
        return dtype_ == DataType::BF16 ? sizeof(std::uint16_t) : sizeof(float);
    }

    // data() is intentionally limited to CPU FP32 tensors so an accidental
    // BF16-as-float or device-as-host access cannot silently corrupt memory.
    // Use raw_data() for CUDA kernels and operator[]/value_at() for converted
    // CPU scalar access.
    float* data() noexcept {
        return device_ == Device::CPU && dtype_ == DataType::FP32
            ? static_cast<float*>(data_ptr_) : nullptr;
    }
    const float* data() const noexcept {
        return device_ == Device::CPU && dtype_ == DataType::FP32
            ? static_cast<const float*>(data_ptr_) : nullptr;
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
        operator float() const { return tensor_->value_at(index_); }
        ElementProxy& operator=(float value) {
            tensor_->set_value(index_, value);
            return *this;
        }
        ElementProxy& operator=(const ElementProxy& other) {
            return *this = static_cast<float>(other);
        }

    private:
        Tensor* tensor_;
        std::size_t index_;
    };

    ElementProxy operator[](std::size_t i) noexcept {
        return ElementProxy(*this, i);
    }
    float operator[](std::size_t i) const { return value_at(i); }
    float value_at(std::size_t i) const;

private:
    struct AliasTag {};
    Tensor(const std::vector<std::size_t>& shape, float* data,
           std::shared_ptr<void> owner, AliasTag);
    Tensor(const std::vector<std::size_t>& shape, void* data,
           std::shared_ptr<void> owner, DataType dtype, AliasTag);
    void compute_strides();
    void set_value(std::size_t i, float value);
    void* data_ptr_ = nullptr;
    Device device_ = Device::CPU;
    DataType dtype_ = DataType::FP32;
    std::size_t total_elements_ = 0;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::shared_ptr<void> owner_;
};
} // namespace stratakv
