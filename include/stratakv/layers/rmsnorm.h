#pragma once

#include "stratakv/tensor.h"

#include <cstddef>

namespace stratakv {
namespace layers {

class RMSNorm {
public:
    explicit RMSNorm(std::size_t hidden_dim,
                     float epsilon = 1.0e-5f);

    RMSNorm(Tensor&& gamma,
            float epsilon = 1.0e-5f);

    RMSNorm(const RMSNorm&)            = delete;
    RMSNorm& operator=(const RMSNorm&) = delete;
    RMSNorm(RMSNorm&&) noexcept        = default;
    RMSNorm& operator=(RMSNorm&&) noexcept = default;
    ~RMSNorm() = default;

    // Normalizes the final dimension. The block passes rank-2
    // [tokens, hidden_dim] tensors, but rank >= 1 is also supported.
    Tensor forward(const Tensor& input) const;

    Tensor&       gamma()       noexcept { return gamma_; }
    const Tensor& gamma() const noexcept { return gamma_; }
    std::size_t hidden_dim() const noexcept { return hidden_dim_; }
    float epsilon() const noexcept { return epsilon_; }

private:
    std::size_t hidden_dim_;
    float epsilon_;
    Tensor gamma_; // [hidden_dim]
};

} // namespace layers
} // namespace stratakv