#include "stratakv/layers/attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stratakv {
namespace layers {

PagedAttention::PagedAttention(std::size_t hidden_dim,
                               std::size_t num_heads,
                               int total_blocks,
                               int block_size)
    : PagedAttention(hidden_dim, num_heads, 0, total_blocks, block_size) {}

PagedAttention::PagedAttention(std::size_t hidden_dim,
                               std::size_t num_heads,
                               std::size_t num_kv_heads,
                               int total_blocks,
                               int block_size)
    : hidden_dim_(hidden_dim)
    , num_heads_(num_heads)
    , num_kv_heads_(num_kv_heads == 0 ? num_heads : num_kv_heads)
    , head_dim_(num_heads == 0 ? 0 : hidden_dim / num_heads)
    , num_groups_(num_kv_heads_ == 0 ? 0 : num_heads / num_kv_heads_)
    , kv_dim_(num_kv_heads_ * head_dim_)
    , total_blocks_(total_blocks)
    , block_size_(block_size)
    , q_proj_(hidden_dim_, hidden_dim_)
    , k_proj_(hidden_dim_, kv_dim_)
    , v_proj_(hidden_dim_, kv_dim_)
    , out_proj_(hidden_dim_, hidden_dim_)
{
    if (hidden_dim == 0 || num_heads == 0) {
        throw std::invalid_argument(
            "PagedAttention: hidden_dim and num_heads must be non-zero");
    }
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument(
            "PagedAttention: hidden_dim must be divisible by num_heads");
    }
    if (num_kv_heads_ == 0 || num_heads_ % num_kv_heads_ != 0) {
        throw std::invalid_argument(
            "PagedAttention: num_kv_heads must divide num_heads");
    }
    if (head_dim_ % 2 != 0) {
        throw std::invalid_argument(
            "PagedAttention: head dimension must be even for RoPE");
    }
    if (total_blocks <= 0 || block_size <= 0) {
        throw std::invalid_argument(
            "PagedAttention: total_blocks and block_size must be positive");
    }

    rope_inv_freq_.resize(head_dim_ / 2u);
    for (std::size_t pair = 0; pair < rope_inv_freq_.size(); ++pair) {
        const double exponent = -2.0 * static_cast<double>(pair) /
            static_cast<double>(head_dim_);
        rope_inv_freq_[pair] = static_cast<float>(
            std::pow(10000.0, exponent));
    }
}

void PagedAttention::ensure_cache(const memory::BlockPool& block_pool) {
    if (block_pool.total_blocks() != total_blocks_ ||
        block_pool.block_size() != block_size_) {
        throw std::invalid_argument(
            "PagedAttention: BlockPool configuration does not match cache");
    }

    if (!key_cache_ && !value_cache_) {
        const std::vector<std::size_t> shape = {
            static_cast<std::size_t>(total_blocks_),
            static_cast<std::size_t>(block_size_),
            kv_dim_};
        // Construct both tensors before publishing either one. This keeps a
        // failed allocation from leaving a half-initialized attention cache.
        Tensor key_cache(shape);
        Tensor value_cache(shape);
        key_cache_ = std::move(key_cache);
        value_cache_ = std::move(value_cache);
    } else if (!key_cache_ || !value_cache_) {
        throw std::runtime_error(
            "PagedAttention: cache is only partially initialized");
    }
}

void PagedAttention::apply_rope(Tensor& queries,
                                Tensor& keys,
                                std::size_t token,
                                std::size_t absolute_position) const {
    float* query_row = queries.data() + token * hidden_dim_;
    float* key_row = keys.data() + token * kv_dim_;

    for (std::size_t head = 0; head < num_heads_; ++head) {
        const std::size_t head_offset = head * head_dim_;
        for (std::size_t pair = 0; pair < rope_inv_freq_.size(); ++pair) {
            const std::size_t first = head_offset + pair;
            const std::size_t second = first + (head_dim_ / 2u);
            const double angle = static_cast<double>(absolute_position) *
                static_cast<double>(rope_inv_freq_[pair]);
            const float cosine = static_cast<float>(std::cos(angle));
            const float sine = static_cast<float>(std::sin(angle));

            const float query_first = query_row[first];
            const float query_second = query_row[second];
            query_row[first] = query_first * cosine - query_second * sine;
            query_row[second] = query_first * sine + query_second * cosine;
        }
    }

    for (std::size_t head = 0; head < num_kv_heads_; ++head) {
        const std::size_t head_offset = head * head_dim_;
        for (std::size_t pair = 0; pair < rope_inv_freq_.size(); ++pair) {
            const std::size_t first = head_offset + pair;
            const std::size_t second = first + (head_dim_ / 2u);
            const double angle = static_cast<double>(absolute_position) *
                static_cast<double>(rope_inv_freq_[pair]);
            const float cosine = static_cast<float>(std::cos(angle));
            const float sine = static_cast<float>(std::sin(angle));
            const float key_first = key_row[first];
            const float key_second = key_row[second];
            key_row[first] = key_first * cosine - key_second * sine;
            key_row[second] = key_first * sine + key_second * cosine;
        }
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
            static_cast<std::size_t>(block_size_) + slot) * kv_dim_;
}

Tensor PagedAttention::forward(const Tensor& hidden_states,
                               int seq_id,
                               memory::PageTable& page_table,
                               memory::BlockPool& block_pool,
                               bool append_tokens) {
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

    // The first layer appends new tokens and advances the shared logical
    // sequence. Later layers use the same mapping, but must write their own
    // private KV tensors at those positions without appending again.
    std::size_t first_position = initial_length;
    if (append_tokens) {
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
    } else {
        if (new_tokens > initial_length) {
            throw std::invalid_argument(
                "PagedAttention::forward: non-appending layer cannot address "
                "tokens beyond the current sequence");
        }
        first_position = initial_length - new_tokens;
    }

    Tensor queries = q_proj_.forward(hidden_states);
    Tensor keys    = k_proj_.forward(hidden_states);
    Tensor values  = v_proj_.forward(hidden_states);
    for (std::size_t token = 0; token < new_tokens; ++token) {
        apply_rope(queries, keys, token, first_position + token);
    }
    Tensor context({new_tokens, hidden_dim_});

    std::vector<unsigned char> visited(
        static_cast<std::size_t>(total_blocks_), 0u);
    last_blocks_visited_ = 0;
    float* key_cache_data = key_cache_->data();
    float* value_cache_data = value_cache_->data();

    for (std::size_t token = 0; token < new_tokens; ++token) {
        const std::size_t logical_position = first_position + token;

        if (append_tokens) {
            // append_token may grow physical_blocks. Do not retain a state
            // reference across this call; obtain a fresh view after appending.
            page_table.append_token(seq_id, block_pool);
        }
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
        std::copy_n(keys.data() + token * kv_dim_,
                    kv_dim_, key_cache_data + offset);
        std::copy_n(values.data() + token * kv_dim_,
                    kv_dim_, value_cache_data + offset);

        // Causal attention. Each logical position is translated through the
        // page table, so logical pages may be physically scattered.
        const float scale = 1.0f / std::sqrt(
            static_cast<float>(head_dim_));
        const float* query_row = queries.data() + token * hidden_dim_;
        float* context_row = context.data() + token * hidden_dim_;

        for (std::size_t head = 0; head < num_heads_; ++head) {
            const std::size_t head_offset = head * head_dim_;
            const std::size_t kv_head = head / num_groups_;
            const std::size_t kv_head_offset = kv_head * head_dim_;
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
                    key_cache_data + position_offset + kv_head_offset;
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
                        value_cache_data + position_offset + kv_head_offset;
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
