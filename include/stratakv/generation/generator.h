#pragma once
#include "stratakv/generation/sampler.h"
#include "stratakv/io/safetensors.h"
#include "stratakv/models/transformer_block.h"
#include <cstddef>
#include <vector>

namespace stratakv::generation {
struct GeneratorConfig {
    std::size_t hidden_dim = 0;
    std::size_t intermediate_dim = 0;
    std::size_t num_heads = 0;
    int total_blocks = 0;
    int block_size = 0;
    std::size_t vocab_size = 0;
    int eos_token_id = -1; // Disabled when -1.
};

class Generator {
public:
    Generator(GeneratorConfig config, io::WeightMap weights);
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens);
    std::size_t free_block_count() const noexcept { return pool_.free_count(); }

private:
    Tensor embed(const int* ids, std::size_t count) const;
    Tensor forward(const Tensor& input);
    // Declaration order ensures validation precedes all internal allocations.
    GeneratorConfig config_;
    models::TransformerBlock block_;
    memory::BlockPool pool_;
    memory::PageTable pages_;
    Sampler sampler_;
    Tensor embeddings_; // [vocab, hidden], owns moved checkpoint tensor.
    layers::RMSNorm final_norm_;
    layers::Linear lm_head_; // GEMM-ready [hidden, vocab].
    bool retry_attention_ = false;
};
} // namespace stratakv::generation
