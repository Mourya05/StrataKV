#pragma once

#include "stratakv/layers/attention.h"
#include "stratakv/layers/linear.h"
#include "stratakv/layers/rmsnorm.h"

#include <cstddef>

namespace stratakv {
namespace models {

class TransformerBlock {
public:
    TransformerBlock(std::size_t hidden_dim,
                     std::size_t intermediate_dim,
                     std::size_t num_heads,
                     int total_blocks,
                     int block_size);

    TransformerBlock(const TransformerBlock&)            = delete;
    TransformerBlock& operator=(const TransformerBlock&) = delete;
    TransformerBlock(TransformerBlock&&) noexcept        = default;
    TransformerBlock& operator=(TransformerBlock&&) noexcept = default;
    ~TransformerBlock() = default;

    Tensor forward(const Tensor& hidden_states,
                   int seq_id,
                   memory::PageTable& page_table,
                   memory::BlockPool& block_pool);

    std::size_t last_attention_blocks_visited() const noexcept {
        return attention_.last_blocks_visited();
    }

    layers::RMSNorm&       input_norm()       noexcept { return norm1_; }
    const layers::RMSNorm& input_norm() const noexcept { return norm1_; }
    layers::PagedAttention&       attention()       noexcept { return attention_; }
    const layers::PagedAttention& attention() const noexcept { return attention_; }
    layers::RMSNorm&       post_attention_norm()       noexcept { return norm2_; }
    const layers::RMSNorm& post_attention_norm() const noexcept { return norm2_; }
    layers::Linear&       ffn_up()       noexcept { return ffn_up_; }
    const layers::Linear& ffn_up() const noexcept { return ffn_up_; }
    layers::Linear&       ffn_down()       noexcept { return ffn_down_; }
    const layers::Linear& ffn_down() const noexcept { return ffn_down_; }

private:
    std::size_t hidden_dim_;
    layers::RMSNorm norm1_;
    layers::PagedAttention attention_;
    layers::RMSNorm norm2_;
    layers::Linear ffn_up_;
    layers::Linear ffn_down_;
};

} // namespace models
} // namespace stratakv