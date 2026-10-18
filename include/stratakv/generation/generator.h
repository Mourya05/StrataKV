#pragma once
#include "stratakv/generation/sampler.h"
#include "stratakv/io/safetensors.h"
#include "stratakv/models/transformer_block.h"
#include <cstddef>
#include <optional>
#include <vector>

namespace stratakv::generation {
struct GeneratorConfig {
    std::size_t hidden_dim = 0;
    std::size_t intermediate_dim = 0;
    std::size_t num_heads = 0;
    std::size_t num_kv_heads = 0; // Zero selects num_heads (standard MHA).
    std::size_t num_layers = 0;
    int total_blocks = 0;
    int block_size = 0;
    std::size_t vocab_size = 0;
    int eos_token_id = -1; // Disabled when -1.
    // Zero keeps existing deterministic generation unless a caller opts into
    // temperature sampling explicitly.
    SamplerConfig sampler{0.0f, 0, 1.0f};

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
        , num_kv_heads(0)
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
        , num_kv_heads(0)
        , num_layers(layers)
        , total_blocks(total)
        , block_size(block)
        , vocab_size(vocab)
        , eos_token_id(eos) {}
    GeneratorConfig(std::size_t hidden,
                    std::size_t intermediate,
                    std::size_t heads,
                    std::size_t kv_heads,
                    std::size_t layers,
                    int total,
                    int block,
                    std::size_t vocab,
                    int eos)
        : hidden_dim(hidden)
        , intermediate_dim(intermediate)
        , num_heads(heads)
        , num_kv_heads(kv_heads)
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
    std::size_t free_block_count() const noexcept;

private:
    Tensor embed(const int* ids, std::size_t count) const;
    void embed_device(int token_id, Tensor& output) const;
    Tensor forward(const Tensor& input);
    // Declaration order ensures validation precedes all internal allocations.
    GeneratorConfig config_;
    std::vector<models::TransformerBlock> blocks_;
    std::vector<memory::BlockPool> pools_;
    std::vector<memory::PageTable> pages_;
    Sampler sampler_;
    Tensor embeddings_; // [vocab, hidden], owns moved checkpoint tensor.
    layers::RMSNorm final_norm_;
    layers::Linear lm_head_; // GEMM-ready [hidden, vocab].
    Device device_ = Device::CPU;
#if defined(STRATAKV_HAS_CUDA)
    std::vector<models::TransformerBlock> parity_blocks_;
    std::vector<memory::BlockPool> parity_pools_;
    std::vector<memory::PageTable> parity_pages_;
    std::optional<Tensor> parity_last_cpu_;
    std::optional<Tensor> gpu_embedding_input_;
    std::optional<Tensor> gpu_final_norm_;
    std::optional<Tensor> gpu_logits_;
    std::optional<Tensor> host_logits_;
#endif
};
} // namespace stratakv::generation
