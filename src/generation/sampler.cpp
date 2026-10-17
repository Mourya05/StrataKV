#include "stratakv/generation/sampler.h"
#include <climits>
#include <cmath>
#include <stdexcept>

namespace stratakv::generation {
int Sampler::sample(const Tensor& logits) const {
    const auto& shape = logits.shape();
    if (!(shape.size() == 1 || (shape.size() == 2 && shape[0] == 1)))
        throw std::invalid_argument("Sampler: expected [vocab] or [1,vocab]");
    const std::size_t vocab = shape.back();
    if (vocab == 0 || vocab > static_cast<std::size_t>(INT_MAX) ||
        logits.num_elements() != vocab || !logits.data())
        throw std::invalid_argument("Sampler: empty or invalid vocabulary/storage");
    std::size_t best = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
        if (!std::isfinite(logits[i]))
            throw std::invalid_argument("Sampler: logits must all be finite");
        if (logits[i] > logits[best]) best = i;
    }
    return static_cast<int>(best); // No softmax is needed for argmax.
}
} // namespace stratakv::generation
