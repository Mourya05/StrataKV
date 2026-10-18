#include "stratakv/layers/rmsnorm.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace stratakv {
namespace layers {

RMSNorm::RMSNorm(std::size_t hidden_dim, float epsilon)
    : hidden_dim_(hidden_dim)
    , epsilon_(epsilon)
    , gamma_({hidden_dim})
{
    if (hidden_dim == 0) {
        throw std::invalid_argument("RMSNorm: hidden_dim must be non-zero");
    }
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        throw std::invalid_argument(
            "RMSNorm: epsilon must be finite and positive");
    }

    // Learned RMSNorm scale starts as identity, not zero.
    for (std::size_t i = 0; i < hidden_dim_; ++i) {
        gamma_[i] = 1.0f;
    }
}

RMSNorm::RMSNorm(Tensor&& gamma, float epsilon)
    : hidden_dim_(gamma.ndim() == 1 ? gamma.shape()[0] : 0)
    , epsilon_(epsilon)
    , gamma_(std::move(gamma))
{
    if (gamma_.ndim() != 1 || gamma_.shape()[0] == 0) {
        throw std::invalid_argument(
            "RMSNorm: gamma must have shape [hidden_dim]");
    }
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        throw std::invalid_argument(
            "RMSNorm: epsilon must be finite and positive");
    }
}

Tensor RMSNorm::forward(const Tensor& input) const {
    if (input.ndim() == 0 || input.shape().back() != hidden_dim_) {
        throw std::invalid_argument(
            "RMSNorm::forward: final input dimension must equal hidden_dim");
    }

    Tensor output(input.shape());
    const std::size_t rows = input.num_elements() / hidden_dim_;
    const float* input_data = input.data();
    float* output_data = output.data();

    for (std::size_t row = 0; row < rows; ++row) {
        const float* input_row = input_data + row * hidden_dim_;
        float* output_row = output_data + row * hidden_dim_;
        double sum_squares = 0.0;
        for (std::size_t col = 0; col < hidden_dim_; ++col) {
            const double value = static_cast<double>(input_row[col]);
            sum_squares += value * value;
        }

        const float mean_square = static_cast<float>(
            sum_squares / static_cast<double>(hidden_dim_));
        const float inverse_rms =
            1.0f / std::sqrt(mean_square + epsilon_);
        for (std::size_t col = 0; col < hidden_dim_; ++col) {
            output_row[col] = input_row[col] * inverse_rms * gamma_[col];
        }
    }

    return output;
}

} // namespace layers
} // namespace stratakv
