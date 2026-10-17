#include "stratakv/layers/attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace stratakv {
namespace layers {

PagedAttention::PagedAttention(std::size_t hidden_dim,
                               std::size_t num_heads,
                               int total_blocks,
                               int block_size)
    : q_proj_(hidden_dim, hidden_dim)
    , k_proj_(hidden_dim, hidden_dim)
    , v_proj_(hidden_dim, hidden_dim)
    , out_proj_(hidden_dim, hidden_dim)
    , hidden_dim_(hidden_dim)
    , num_heads_(num_heads)
    , head_dim_(num_heads == 0 ? 0 : hidden_dim / num_heads)
    , total_blocks_(total_blocks)
    , block_size_(block_size)
{
    if (hidden_dim == 0 || num_heads == 0) {
        throw std::invalid_argument(
            "PagedAttention: hidden_dim and num_heads must be non-zero");
    }
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument(
            "PagedAttention: hidden_dim must be divisible by num_heads");
    }
    if (total_blocks <= 0 || block_size <= 0) {
        throw std::invalid_argument(
            "PagedAttention: total_blocks and block_size must be positive");
    }
}

void PagedAttention::ensure_cache(const memory::BlockPool& block_pool) {
    if (block_pool.total_blocks() != total_blocks_ ||
        block_pool.block_size() != block_size_) {
        throw std::invalid_argument(
            "PagedAttention: BlockPool configuration does not match cache");
    }

    if (!key_cache_) {
        const std::vector<std::size_t> shape = {
            static_cast<std::size_t>(total_blocks_),
            static_cast<std::size_t>(block_size_),
            hidden_dim_};
        key_cache_.emplace(shape);
        value_cache_.emplace(shape);
    }
}

std::size_t PagedAttention::cache_offset(int physical_block,
                                         std::size_t slot) const {
    if (physical_block < 0 || physical_block >= total_blocks_) {
        throw std::runtime_error(
            "PagedAttention: invalid physical block " +
            std::to_string(physical_block));
    }
    if (slot >= static_cast<std::size_t>(block_size_)) {
        throw std::runtime_error(
            "PagedAttention: KV slot is outside the physical block");
    }

    return (static_cast<std::size_t>(physical_block) *
            static_cast<std::size_t>(block_size_) + slot) * hidden_dim_;
}

Tensor PagedAttention::forward(const Tensor& hidden_states,
                               int seq_id,
                               memory::PageTable& page_table,
                               memory::BlockPool& block_pool) {
    if (hidden_states.ndim() != 2 ||
        hidden_states.shape()[1] != hidden_dim_) {
        throw std::invalid_argument(
            "PagedAttention::forward: hidden_states must be "
            "[new_tokens, hidden_dim]");
    }
    if (!page_table.has_sequence(seq_id)) {
        throw std::runtime_error(
            "PagedAttention::forward: seq_id is not registered");
    }

    ensure_cache(block_pool);

    const std::size_t new_tokens = hidden_states.shape()[0];
    const memory::SequenceState& initial_state = page_table.get_state(seq_id);
    if (initial_state.sequence_length < 0) {
        throw std::runtime_error(
            "PagedAttention::forward: negative sequence length");
    }
    const std::size_t initial_length =
        static_cast<std::size_t>(initial_state.sequence_length);
    if (new_tokens >
        static_cast<std::size_t>(std::numeric_limits<int>::max()) -
            initial_length) {
        throw std::overflow_error(
            "PagedAttention::forward: sequence length exceeds int range");
    }

    // Preflight before mutating the page table. This prevents an avoidable
    // partial append when the pool cannot hold the requested chunk.
    const std::size_t total_tokens = initial_length + new_tokens;
    const std::size_t required_blocks =
        (total_tokens + static_cast<std::size_t>(block_size_) - 1u) /
        static_cast<std::size_t>(block_size_);
    const std::size_t allocated_blocks = initial_state.physical_blocks.size();
    const std::size_t blocks_needed =
        required_blocks > allocated_blocks
            ? required_blocks - allocated_blocks
            : 0u;
    if (blocks_needed > block_pool.free_count()) {
        throw std::runtime_error(
            "PagedAttention::forward: insufficient KV blocks (OOM)");
    }

    Tensor queries = q_proj_.forward(hidden_states);
    Tensor keys    = k_proj_.forward(hidden_states);
    Tensor values  = v_proj_.forward(hidden_states);
    Tensor context({new_tokens, hidden_dim_});

    std::vector<unsigned char> visited(
        static_cast<std::size_t>(total_blocks_), 0u);
    last_blocks_visited_ = 0;
    float* key_cache_data = key_cache_->data();
    float* value_cache_data = value_cache_->data();

    for (std::size_t token = 0; token < new_tokens; ++token) {
        const std::size_t logical_position = initial_length + token;

        // append_token may grow physical_blocks. Do not retain a state
        // reference across this call; obtain a fresh view after appending.
        page_table.append_token(seq_id, block_pool);
        const memory::SequenceState& state = page_table.get_state(seq_id);
        const std::size_t page_index =
            logical_position / static_cast<std::size_t>(block_size_);
        const std::size_t slot =
            logical_position % static_cast<std::size_t>(block_size_);
        if (page_index >= state.physical_blocks.size()) {
            throw std::runtime_error(
                "PagedAttention: no page for appended token");
        }

        const int physical_block = state.physical_blocks.at(page_index);
        const std::size_t offset = cache_offset(physical_block, slot);
        std::copy_n(keys.data() + token * hidden_dim_,
                    hidden_dim_, key_cache_data + offset);
        std::copy_n(values.data() + token * hidden_dim_,
                    hidden_dim_, value_cache_data + offset);

        // Causal attention. Each logical position is translated through the
        // page table, so logical pages may be physically scattered.
        const float scale = 1.0f / std::sqrt(
            static_cast<float>(head_dim_));
        const float* query_row = queries.data() + token * hidden_dim_;
        float* context_row = context.data() + token * hidden_dim_;

        for (std::size_t head = 0; head < num_heads_; ++head) {
            const std::size_t head_offset = head * head_dim_;
            std::vector<float> scores(logical_position + 1u);
            float max_score = -std::numeric_limits<float>::infinity();

            for (std::size_t position = 0;
                 position <= logical_position;
                 ++position) {
                const std::size_t position_page =
                    position / static_cast<std::size_t>(block_size_);
                const std::size_t position_slot =
                    position % static_cast<std::size_t>(block_size_);
                if (position_page >= state.physical_blocks.size()) {
                    throw std::runtime_error(
                        "PagedAttention: no physical page for position");
                }

                const int position_block =
                    state.physical_blocks.at(position_page);
                const std::size_t position_offset =
                    cache_offset(position_block, position_slot);
                const std::size_t visited_index =
                    static_cast<std::size_t>(position_block);
                if (visited.at(visited_index) == 0u) {
                    visited[visited_index] = 1u;
                    ++last_blocks_visited_;
                }

                const float* key =
                    key_cache_data + position_offset + head_offset;
                float dot = 0.0f;
                for (std::size_t d = 0; d < head_dim_; ++d) {
                    dot += query_row[head_offset + d] * key[d];
                }
                scores[position] = dot * scale;
                max_score = std::max(max_score, scores[position]);
            }

            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - max_score);
                denominator += score;
            }
            if (!(denominator > 0.0f) || !std::isfinite(denominator)) {
                throw std::runtime_error(
                    "PagedAttention: invalid softmax denominator");
            }

            for (std::size_t d = 0; d < head_dim_; ++d) {
                float accumulated = 0.0f;
                for (std::size_t position = 0;
                     position <= logical_position;
                     ++position) {
                    const std::size_t position_page =
                        position / static_cast<std::size_t>(block_size_);
                    const std::size_t position_slot =
                        position % static_cast<std::size_t>(block_size_);
                    const int position_block =
                        state.physical_blocks.at(position_page);
                    const std::size_t position_offset =
                        cache_offset(position_block, position_slot);
                    const float* value =
                        value_cache_data + position_offset + head_offset;
                    accumulated +=
                        (scores[position] / denominator) * value[d];
                }
                context_row[head_offset + d] = accumulated;
            }
        }
    }

    return out_proj_.forward(context);
}

} // namespace layers
} // namespace stratakv