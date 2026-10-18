#pragma once

#include "stratakv/layers/linear.h"
#include "stratakv/memory/block_pool.h"
#include "stratakv/memory/page_table.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace stratakv {
namespace layers {

class PagedAttention {
public:
    PagedAttention(std::size_t hidden_dim,
                   std::size_t num_heads,
                   int total_blocks,
                   int block_size,
                   Device device = Device::CPU);
    PagedAttention(std::size_t hidden_dim,
                   std::size_t num_heads,
                   std::size_t num_kv_heads,
                   int total_blocks,
                   int block_size,
                   Device device = Device::CPU);

    PagedAttention(const PagedAttention&)            = delete;
    PagedAttention& operator=(const PagedAttention&) = delete;
    PagedAttention(PagedAttention&&) noexcept        = default;
    PagedAttention& operator=(PagedAttention&&) noexcept = default;
    ~PagedAttention() = default;

    // hidden_states is [new_tokens, hidden_dim]. Each row is appended to the
    // sequence and then attends causally to all valid sequence positions.
    // Device::GPU reserves device KV storage, but full GPU attention kernels
    // are not yet part of this CPU attention implementation.
    Tensor forward(const Tensor& hidden_states,
                   int seq_id,
                   memory::PageTable& page_table,
                   memory::BlockPool& block_pool,
                   bool append_tokens = true);

    // Number of distinct physical blocks read by the last forward pass.
    std::size_t last_blocks_visited() const noexcept {
        return last_blocks_visited_;
    }

    Linear&       query_projection()       noexcept { return q_proj_; }
    const Linear& query_projection() const noexcept { return q_proj_; }
    Linear&       key_projection()        noexcept { return k_proj_; }
    const Linear& key_projection()  const noexcept { return k_proj_; }
    Linear&       value_projection()      noexcept { return v_proj_; }
    const Linear& value_projection() const noexcept { return v_proj_; }
    Linear&       output_projection()     noexcept { return out_proj_; }
    const Linear& output_projection() const noexcept { return out_proj_; }
    Device device() const noexcept { return device_; }

private:
    void ensure_cache(const memory::BlockPool& block_pool);
    void apply_rope(Tensor& queries,
                    Tensor& keys,
                    std::size_t token,
                    std::size_t absolute_position) const;
    std::size_t cache_offset(int physical_block,
                             std::size_t slot) const;

    std::size_t hidden_dim_;
    std::size_t num_heads_;
    std::size_t num_kv_heads_;
    std::size_t head_dim_;
    std::size_t num_groups_;
    std::size_t kv_dim_;
    int total_blocks_;
    int block_size_;
    Device device_;
    std::vector<float> rope_inv_freq_;

    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear out_proj_;

    // [total_blocks, block_size, kv_dim]. A physical block ID is the first
    // index; the page table supplies that ID for each logical position.
    std::optional<Tensor> key_cache_;
    std::optional<Tensor> value_cache_;
    std::size_t last_blocks_visited_ = 0;
};

} // namespace layers
} // namespace stratakv
