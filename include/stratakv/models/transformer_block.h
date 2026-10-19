#pragma once

#include "stratakv/layers/attention.h"
#include "stratakv/layers/linear.h"
#include "stratakv/layers/rmsnorm.h"

#include <cstddef>
#include <optional>

namespace stratakv {
namespace models {

class TransformerBlock {
public:
    TransformerBlock(std::size_t hidden_dim,
                     std::size_t intermediate_dim,
                     std::size_t num_heads,
                     int total_blocks,
                     int block_size,
                     Device device = Device::CPU);
    TransformerBlock(std::size_t hidden_dim,
                     std::size_t intermediate_dim,
                     std::size_t num_heads,
                     std::size_t num_kv_heads,
                     int total_blocks,
                     int block_size,
                     Device device = Device::CPU);

    TransformerBlock(const TransformerBlock&)            = delete;
    TransformerBlock& operator=(const TransformerBlock&) = delete;
    TransformerBlock(TransformerBlock&&) noexcept        = default;
    TransformerBlock& operator=(TransformerBlock&&) noexcept = default;
    ~TransformerBlock() = default;

    // Move every parameter owned by the block, including attention/FFN
    // projections and both RMSNorm gamma tensors, to the requested device.
    void to(Device device);
    void set_parity_diagnostic(bool enabled) noexcept {
        parity_diagnostic_ = enabled;
    }

    Tensor forward(const Tensor& hidden_states,
                   int seq_id,
                   memory::PageTable& page_table,
                   memory::BlockPool& block_pool,
                   bool append_tokens = true);
    void forward_into(const Tensor& hidden_states, Tensor& output,
                      int seq_id, memory::PageTable& page_table,
                      memory::BlockPool& block_pool,
                      bool append_tokens = true);

    std::size_t last_attention_blocks_visited() const noexcept {
        return attention_.last_blocks_visited();
    }

    layers::RMSNorm&       input_norm()       noexcept { return norm1_; }
    const layers::RMSNorm& input_norm() const noexcept { return norm1_; }
    layers::PagedAttention&       attention()       noexcept { return attention_; }
    const layers::PagedAttention& attention() const noexcept { return attention_; }
    layers::RMSNorm&       post_attention_norm()       noexcept { return norm2_; }
    const layers::RMSNorm& post_attention_norm() const noexcept { return norm2_; }
    layers::Linear&       gate_projection()       noexcept { return gate_proj_; }
    const layers::Linear& gate_projection() const noexcept { return gate_proj_; }
    layers::Linear&       up_projection()       noexcept { return up_proj_; }
    const layers::Linear& up_projection() const noexcept { return up_proj_; }
    layers::Linear&       down_projection()       noexcept { return down_proj_; }
    const layers::Linear& down_projection() const noexcept { return down_proj_; }

    // Short aliases mirror the names used by the Llama checkpoint schema.
    layers::Linear&       gate_proj()       noexcept { return gate_proj_; }
    const layers::Linear& gate_proj() const noexcept { return gate_proj_; }
    layers::Linear&       up_proj()       noexcept { return up_proj_; }
    const layers::Linear& up_proj() const noexcept { return up_proj_; }
    layers::Linear&       down_proj()       noexcept { return down_proj_; }
    const layers::Linear& down_proj() const noexcept { return down_proj_; }

private:
    std::size_t hidden_dim_;
    layers::RMSNorm norm1_;
    layers::PagedAttention attention_;
    layers::RMSNorm norm2_;
    layers::Linear gate_proj_;
    layers::Linear up_proj_;
    layers::Linear down_proj_;
    bool parity_diagnostic_ = false;
#if defined(STRATAKV_HAS_CUDA)
    // These workspaces are allocated once by to(Device::GPU) and reused by
    // forward_into during token generation.  The block is intentionally not
    // safe for concurrent requests; each in-flight request needs its own
    // TransformerBlock instance (or workspace set).
    std::optional<Tensor> normalized_input_workspace_;
    std::optional<Tensor> query_workspace_;
    std::optional<Tensor> key_workspace_;
    std::optional<Tensor> value_workspace_;
    std::optional<Tensor> cos_workspace_;
    std::optional<Tensor> sin_workspace_;
    std::optional<Tensor> context_workspace_;
    std::optional<Tensor> normalized_residual_workspace_;
    std::optional<Tensor> gate_workspace_;
    std::optional<Tensor> up_workspace_;
    std::optional<Tensor> down_workspace_;
#endif
};

} // namespace models
} // namespace stratakv
