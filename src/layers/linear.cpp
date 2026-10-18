#include "stratakv/layers/linear.h"

#include "stratakv/ops.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stratakv {
namespace layers {
namespace {

std::size_t checked_weight_rows(const Tensor& weight) {
    if (weight.ndim() != 2) {
        throw std::invalid_argument("Linear: weight must be rank-2");
    }
    return weight.shape()[0];
}

std::size_t checked_weight_cols(const Tensor& weight) {
    if (weight.ndim() != 2) {
        throw std::invalid_argument("Linear: weight must be rank-2");
    }
    return weight.shape()[1];
}

} // namespace

Linear::Linear(std::size_t input_dim,
               std::size_t output_dim,
               bool use_bias)
    : input_dim_(input_dim)
    , output_dim_(output_dim)
    , weight_({input_dim, output_dim})
{
    if (input_dim == 0 || output_dim == 0) {
        throw std::invalid_argument(
            "Linear: input_dim and output_dim must be non-zero");
    }
    if (use_bias) {
        bias_.emplace(std::vector<std::size_t>{output_dim});
    }
}

Linear::Linear(Tensor&& weight_transposed,
               std::optional<Tensor> bias)
    : input_dim_(checked_weight_rows(weight_transposed))
    , output_dim_(checked_weight_cols(weight_transposed))
    , weight_(std::move(weight_transposed))
    , bias_(std::move(bias))
{
    if (input_dim_ == 0 || output_dim_ == 0) {
        throw std::invalid_argument(
            "Linear: weight dimensions must be non-zero");
    }
    if (bias_ &&
        (bias_->ndim() != 1 || bias_->shape()[0] != output_dim_)) {
        throw std::invalid_argument(
            "Linear: bias must have shape [output_dim]");
    }
}

Tensor* Linear::bias() noexcept {
    return bias_ ? &*bias_ : nullptr;
}

const Tensor* Linear::bias() const noexcept {
    return bias_ ? &*bias_ : nullptr;
}

void Linear::set_weight(Tensor&& weight_transposed) {
    if (weight_transposed.ndim() != 2 ||
        weight_transposed.shape()[0] != input_dim_ ||
        weight_transposed.shape()[1] != output_dim_ ||
        weight_transposed.num_elements() != input_dim_ * output_dim_) {
        throw std::invalid_argument(
            "Linear::set_weight: weight shape does not match layer");
    }
    weight_ = std::move(weight_transposed);
}

void Linear::to(Device device) {
    weight_ = weight_.to(device);
    if (bias_)
        *bias_ = bias_->to(device);
}

Tensor Linear::forward(const Tensor& input) const {
    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "Linear::forward: input must have shape [tokens, input_dim]");
    }
    if (input.shape()[1] != input_dim_) {
        throw std::invalid_argument(
            "Linear::forward: input feature dimension is " +
            std::to_string(input.shape()[1]) + ", expected " +
            std::to_string(input_dim_));
    }

    if (input.device() != weight_.device()) {
        throw std::invalid_argument(
            "Linear::forward: input and weight must share a device");
    }
    Tensor output({input.shape()[0], output_dim_}, DataType::FP32,
                  input.device());
    forward_into(input, output);
    return output;
}

void Linear::forward_into(const Tensor& input, Tensor& output) const {
    if (input.ndim() != 2 || input.shape()[1] != input_dim_ ||
        output.ndim() != 2 ||
        output.shape() != std::vector<std::size_t>{
            input.shape()[0], output_dim_} ||
        output.dtype() != DataType::FP32 ||
        output.device() != input.device()) {
        throw std::invalid_argument(
            "Linear::forward_into: input/output shape or device mismatch");
    }

#if defined(STRATAKV_HAS_CUDA)
    if (input.device() == Device::GPU)
        matmul_cuda_tiled(input, weight_, output);
    else
        matmul_cpu_avx2(input, weight_, output);
#else
    if (input.device() == Device::GPU) {
        throw std::runtime_error(
            "Linear::forward: GPU tensors require CUDA support");
    }
    matmul_cpu_avx2(input, weight_, output);
#endif

    if (bias_) {
#if defined(STRATAKV_HAS_CUDA)
        if (output.device() == Device::GPU) {
            if (bias_->device() != Device::GPU)
                throw std::invalid_argument(
                    "Linear::forward: GPU bias must be GPU-resident");
            add_bias_cuda(output, *bias_);
            return;
        }
#else
        if (output.device() == Device::GPU)
            throw std::runtime_error(
                "Linear::forward: GPU bias requires CUDA support");
#endif
        const float* bias_data = bias_->data();
        float* output_data = output.data();
        const std::size_t rows = input.shape()[0];
        for (std::size_t row = 0; row < rows; ++row) {
            float* output_row = output_data + row * output_dim_;
            for (std::size_t col = 0; col < output_dim_; ++col) {
                output_row[col] += bias_data[col];
            }
        }
    }

}

} // namespace layers
} // namespace stratakv
