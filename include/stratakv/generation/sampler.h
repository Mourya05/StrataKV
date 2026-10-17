#pragma once
#include "stratakv/tensor.h"

namespace stratakv::generation {
class Sampler {
public:
    // Greedy logits argmax; equal maxima select the lowest token ID.
    int sample(const Tensor& logits) const;
};
} // namespace stratakv::generation
