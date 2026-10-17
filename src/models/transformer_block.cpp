#include "stratakv/models/transformer_block.h"

#include <cmath>
#include <stdexcept>

namespace stratakv {
namespace models {

TransformerBlock::TransformerBlock(std::size_t hidden_dim,
                                   std::size_t intermediate_dim,
                                   std::size_t num_heads,
                                   int total_blocks,
                                   int block_size)
    : TransformerBlock(hidden_dim, intermediate_dim, num_heads, 0,
                       total_blocks, block_size) {}

TransformerBlock::TransformerBlock(std::size_t hidden_dim,
                                   std::size_t intermediate_dim,
                                   std::size_t num_heads,
                                   std::size_t num_kv_heads,
                                   int total_blocks,
                                   int block_size)
    : hidden_dim_(hidden_dim)
    , norm1_(hidden_dim)
    , attention_(hidden_dim, num_heads, num_kv_heads, total_blocks, block_size)
    , norm2_(hidden_dim)
    , gate_proj_(hidden_dim, intermediate_dim)
    , up_proj_(hidden_dim, intermediate_dim)
    , down_proj_(intermediate_dim, hidden_dim)
{
    if (hidden_dim == 0 || intermediate_dim == 0) {
        throw std::invalid_argument(
            "TransformerBlock: feature dimensions must be non-zero");
    }
}

Tensor TransformerBlock::forward(const Tensor& hidden_states,
                                 int seq_id,
                                 memory::PageTable& page_table,
                                 memory::BlockPool& block_pool,
                                 bool append_tokens) {
    if (hidden_states.ndim() != 2 ||
        hidden_states.shape()[1] != hidden_dim_) {
        throw std::invalid_argument(
            "TransformerBlock::forward: hidden_states must be "
            "[tokens, hidden_dim]");
    }

    // H = X + Attention(RMSNorm(X))
    Tensor normalized_input = norm1_.forward(hidden_states);
    Tensor attention_output = attention_.forward(
        normalized_input, seq_id, page_table, block_pool, append_tokens);
    Tensor attention_residual({hidden_states.shape()[0], hidden_dim_});
    const std::size_t element_count = hidden_states.num_elements();
    for (std::size_t i = 0; i < element_count; ++i) {
        attention_residual[i] = hidden_states[i] + attention_output[i];
    }

    // X_out = H + down_proj(SiLU(gate_proj(RMSNorm(H))) * up_proj(RMSNorm(H))).
    Tensor normalized_residual = norm2_.forward(attention_residual);
    Tensor gate = gate_proj_.forward(normalized_residual);
    Tensor up = up_proj_.forward(normalized_residual);
    for (std::size_t i = 0; i < gate.num_elements(); ++i) {
        const float value = gate[i];
        // This form avoids exp overflow for large negative gate values while
        // remaining mathematically equivalent to x / (1 + exp(-x)).
        const float sigmoid = value >= 0.0f
            ? 1.0f / (1.0f + std::exp(-value))
            : std::exp(value) / (1.0f + std::exp(value));
        gate[i] = value * sigmoid * up[i];
    }

    Tensor ffn_output = down_proj_.forward(gate);
    Tensor output({hidden_states.shape()[0], hidden_dim_});
    for (std::size_t i = 0; i < element_count; ++i) {
        output[i] = attention_residual[i] + ffn_output[i];
    }

    return output;
}

} // namespace models
} // namespace stratakv
