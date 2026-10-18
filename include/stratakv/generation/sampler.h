#pragma once
#include "stratakv/tensor.h"
#include <cstdint>
#include <random>

namespace stratakv::generation {

struct SamplerConfig {
    // A zero temperature keeps the legacy deterministic greedy behavior.
    // Positive temperatures enable categorical sampling after filtering.
    float temperature = 1.0f;
    // Zero disables Top-K filtering. Values larger than the vocabulary are
    // clamped at sampling time.
    int top_k = 0;
    // Nucleus threshold in (0, 1]. A value of 1 disables Top-P filtering.
    float top_p = 1.0f;
};

class Sampler {
public:
    Sampler();
    explicit Sampler(SamplerConfig config);
    Sampler(SamplerConfig config, std::uint32_t seed);

    // Greedy logits argmax when temperature is zero; otherwise apply
    // temperature, Top-K, Top-P, and sample the remaining categorical mass.
    int sample(const Tensor& logits) const;

    const SamplerConfig& config() const noexcept { return config_; }

private:
    SamplerConfig config_;
    mutable std::mt19937 rng_;
};
} // namespace stratakv::generation
