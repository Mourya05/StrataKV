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
    std::size_t num_layers = 0;
    int total_blocks = 0;
    int block_size = 0;
    std::size_t vocab_size = 0;
    int eos_token_id = -1; // Disabled when -1.

    // Keep the Phase 5 positional form source-compatible as a one-layer
    // configuration while allowing the full Phase 6 form below.
    GeneratorConfig() = default;
    GeneratorConfig(std::size_t hidden,
                    std::size_t intermediate,
                    std::size_t heads,
                    int total,
                    int block,
                    std::size_t vocab,
                    int eos)
        : hidden_dim(hidden)
        , intermediate_dim(intermediate)
        , num_heads(heads)
        , num_layers(1)
        , total_blocks(total)
        , block_size(block)
        , vocab_size(vocab)
        , eos_token_id(eos) {}
    GeneratorConfig(std::size_t hidden,
                    std::size_t intermediate,
                    std::size_t heads,
                    std::size_t layers,
                    int total,
                    int block,
                    std::size_t vocab,
                    int eos)
        : hidden_dim(hidden)
        , intermediate_dim(intermediate)
        , num_heads(heads)
        , num_layers(layers)
        , total_blocks(total)
        , block_size(block)
        , vocab_size(vocab)
        , eos_token_id(eos) {}
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
    std::vector<models::TransformerBlock> blocks_;
    memory::BlockPool pool_;
    memory::PageTable pages_;
    Sampler sampler_;
    Tensor embeddings_; // [vocab, hidden], owns moved checkpoint tensor.
    layers::RMSNorm final_norm_;
    layers::Linear lm_head_; // GEMM-ready [hidden, vocab].
};
} // namespace stratakv::generation
