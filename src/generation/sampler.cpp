#include "stratakv/generation/sampler.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace stratakv::generation {

namespace {
void validate_config(const SamplerConfig& config) {
    if (!std::isfinite(config.temperature) || config.temperature < 0.0f)
        throw std::invalid_argument(
            "Sampler: temperature must be finite and nonnegative");
    if (config.top_k < 0)
        throw std::invalid_argument("Sampler: top_k must be nonnegative");
    if (!std::isfinite(config.top_p) || config.top_p <= 0.0f ||
        config.top_p > 1.0f)
        throw std::invalid_argument("Sampler: top_p must be in (0, 1]");
}

std::uint32_t random_seed() {
    std::random_device device;
    return device();
}
} // namespace

Sampler::Sampler()
    : config_{0.0f, 0, 1.0f}, rng_(random_seed()) {}

Sampler::Sampler(SamplerConfig config)
    : config_(config), rng_(random_seed()) {
    validate_config(config_);
}

Sampler::Sampler(SamplerConfig config, std::uint32_t seed)
    : config_(config), rng_(seed) {
    validate_config(config_);
}

int Sampler::sample(const Tensor& logits) const {
    const auto& shape = logits.shape();
    if (!(shape.size() == 1 || (shape.size() == 2 && shape[0] == 1)))
        throw std::invalid_argument("Sampler: expected [vocab] or [1,vocab]");
    const std::size_t vocab = shape.back();
    if (vocab == 0 || vocab > static_cast<std::size_t>(INT_MAX) ||
        logits.num_elements() != vocab || !logits.raw_data())
        throw std::invalid_argument("Sampler: empty or invalid vocabulary/storage");
    std::vector<float> values(vocab);
    for (std::size_t i = 0; i < vocab; ++i) {
        values[i] = logits[i];
        if (!std::isfinite(values[i]))
            throw std::invalid_argument("Sampler: logits must all be finite");
    }

    std::size_t best = 0;
    for (std::size_t i = 1; i < vocab; ++i)
        if (values[i] > values[best]) best = i;
    if (config_.temperature == 0.0f)
        return static_cast<int>(best); // Compatibility greedy mode.

    for (float& value : values) {
        value /= config_.temperature;
        if (!std::isfinite(value))
            throw std::invalid_argument("Sampler: scaled logits are not finite");
    }

    std::vector<std::size_t> order(vocab);
    std::iota(order.begin(), order.end(), std::size_t{0});
    const auto descending = [&values](std::size_t left, std::size_t right) {
        if (values[left] != values[right]) return values[left] > values[right];
        return left < right;
    };
    std::sort(order.begin(), order.end(), descending);

    std::vector<unsigned char> allowed(vocab, 1u);
    if (config_.top_k > 0 &&
        static_cast<std::size_t>(config_.top_k) < vocab) {
        for (std::size_t rank = static_cast<std::size_t>(config_.top_k);
             rank < vocab; ++rank)
            allowed[order[rank]] = 0u;
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < vocab; ++i)
        if (allowed[i]) max_logit = (std::max)(max_logit, values[i]);

    std::vector<float> weights(vocab, 0.0f);
    float total = 0.0f;
    for (std::size_t i = 0; i < vocab; ++i) {
        if (!allowed[i]) continue;
        weights[i] = std::exp(values[i] - max_logit);
        total += weights[i];
    }
    if (!(total > 0.0f) || !std::isfinite(total))
        throw std::runtime_error("Sampler: invalid softmax distribution");

    if (config_.top_p < 1.0f) {
        std::vector<std::size_t> probability_order;
        probability_order.reserve(vocab);
        for (std::size_t i = 0; i < vocab; ++i)
            if (allowed[i]) probability_order.push_back(i);
        std::sort(probability_order.begin(), probability_order.end(),
                  [&weights](std::size_t left, std::size_t right) {
                      if (weights[left] != weights[right])
                          return weights[left] > weights[right];
                      return left < right;
                  });

        float cumulative = 0.0f;
        for (std::size_t index : probability_order) {
            cumulative += weights[index] / total;
            // The token that crosses top_p is excluded, matching the phase
            // specification's "cumsum > top_p" masking rule.
            if (cumulative > config_.top_p) allowed[index] = 0u;
        }
        if (std::none_of(allowed.begin(), allowed.end(),
                         [](unsigned char value) { return value != 0u; })) {
            allowed[order.front()] = 1u;
        }
    }

    std::fill(weights.begin(), weights.end(), 0.0f);
    total = 0.0f;
    for (std::size_t i = 0; i < vocab; ++i) {
        if (!allowed[i]) continue;
        weights[i] = std::exp(values[i] - max_logit);
        total += weights[i];
    }
    if (!(total > 0.0f) || !std::isfinite(total))
        throw std::runtime_error("Sampler: invalid filtered distribution");
    for (float& weight : weights) weight /= total;

    std::discrete_distribution<std::size_t> distribution(
        weights.begin(), weights.end());
    return static_cast<int>(distribution(rng_));
}
} // namespace stratakv::generation
