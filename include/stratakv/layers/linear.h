#pragma once

#include "stratakv/tensor.h"

#include <cstddef>
#include <optional>

namespace stratakv {
namespace layers {

// Y = X * W^T + bias.
// X is [tokens, input_dim]. The stored physical weight is [input_dim,
// output_dim], which is W^T and can be passed directly to GEMM as B.
class Linear {
public:
    Linear(std::size_t input_dim,
           std::size_t output_dim,
           bool use_bias = false);

    // Construct from an already-transposed, GEMM-ready weight tensor.
    // weight_transposed must have shape [input_dim, output_dim].
    Linear(Tensor&& weight_transposed,
           std::optional<Tensor> bias = std::nullopt);

    Linear(const Linear&)            = delete;
    Linear& operator=(const Linear&) = delete;
    Linear(Linear&&) noexcept        = default;
    Linear& operator=(Linear&&) noexcept = default;
    ~Linear() = default;

    Tensor forward(const Tensor& input) const;

    // The returned weight is the physical transposed/GEMM-ready tensor.
    Tensor&       weight()       noexcept { return weight_; }
    const Tensor& weight() const noexcept { return weight_; }

    Tensor*       bias()       noexcept;
    const Tensor* bias() const noexcept;

    std::size_t input_dim()  const noexcept { return input_dim_; }
    std::size_t output_dim() const noexcept { return output_dim_; }

private:
    std::size_t input_dim_;
    std::size_t output_dim_;
    Tensor weight_;              // [input_dim, output_dim] = W^T
    std::optional<Tensor> bias_; // [output_dim], when enabled
};

} // namespace layers
} // namespace stratakv