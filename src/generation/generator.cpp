#include "stratakv/generation/generator.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace stratakv::generation {
namespace {
std::size_t checked_elements(std::initializer_list<std::size_t> dims) {
    const auto byte_limit = (std::min)(
        (std::numeric_limits<std::size_t>::max)() - 63u,
        static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()));
    const auto limit = byte_limit / sizeof(float);
    std::size_t n = 1;
    for (auto d : dims) {
        if (d == 0 || n > limit / d)
            throw std::overflow_error("Generator: invalid/overflowing tensor size");
        n *= d;
    }
    return n;
}

GeneratorConfig validate_config(GeneratorConfig c) {
    if (c.hidden_dim == 0 || c.intermediate_dim == 0 || c.num_heads == 0 ||
        c.vocab_size == 0 || c.vocab_size > static_cast<std::size_t>(INT_MAX) ||
        c.total_blocks <= 0 || c.block_size <= 0 ||
        c.hidden_dim % c.num_heads != 0 || c.eos_token_id < -1 ||
        (c.eos_token_id >= 0 &&
         static_cast<std::size_t>(c.eos_token_id) >= c.vocab_size))
        throw std::invalid_argument("Generator: invalid configuration");
    checked_elements({c.hidden_dim, c.hidden_dim});
    checked_elements({c.hidden_dim, c.intermediate_dim});
    checked_elements({c.vocab_size, c.hidden_dim});
    checked_elements({static_cast<std::size_t>(c.total_blocks),
                      static_cast<std::size_t>(c.block_size), c.hidden_dim});
    return c;
}

Tensor take(io::WeightMap& weights, const char* key,
            std::initializer_list<std::size_t> expected) {
    auto it = weights.find(key); // Never operator[]: Tensor has no default ctor.
    if (it == weights.end())
        throw std::invalid_argument(std::string("Missing weight: ") + key);
    const Tensor& source = it->second;
    if (source.shape() != std::vector<std::size_t>(expected) ||
        source.num_elements() != checked_elements(expected) || !source.data())
        throw std::invalid_argument(std::string("Wrong shape/storage: ") + key);
    for (std::size_t i = 0; i < source.num_elements(); ++i)
        if (!std::isfinite(source[i]))
            throw std::invalid_argument(std::string("Non-finite weight: ") + key);
    Tensor result = std::move(it->second);
    weights.erase(it);
    return result;
}

void load_linear(io::WeightMap& weights, const char* key, layers::Linear& layer) {
    const auto in = layer.input_dim(), out = layer.output_dim();
    Tensor disk = take(weights, key, {out, in});
    Tensor& dest = layer.weight(); // Transpose into existing allocation once.
    for (std::size_t o = 0; o < out; ++o)
        for (std::size_t i = 0; i < in; ++i)
            dest[i * out + o] = disk[o * in + i];
    if (Tensor* bias = layer.bias())
        std::fill_n(bias->data(), bias->num_elements(), 0.0f);
}

// Existing PageTable can pop a pool block before vector/map growth throws.
// Snapshots recover even unregistered blocks that free_sequence cannot see.
class SequenceGuard {
public:
    SequenceGuard(memory::PageTable& pages, memory::BlockPool& pool)
        : pages_(pages), pool_(pool), saved_pages_(pages), saved_pool_(pool) {}
    SequenceGuard(const SequenceGuard&) = delete;
    SequenceGuard& operator=(const SequenceGuard&) = delete;
    ~SequenceGuard() noexcept {
        try {
            if (pages_.has_sequence(0)) pages_.free_sequence(0, pool_);
        } catch (...) { /* Snapshot rollback also handles cleanup failure. */ }
        pages_ = std::move(saved_pages_);
        pool_ = std::move(saved_pool_);
    }
private:
    memory::PageTable& pages_;
    memory::BlockPool& pool_;
    memory::PageTable saved_pages_;
    memory::BlockPool saved_pool_;
};
static_assert(std::is_nothrow_move_assignable_v<memory::PageTable> &&
              std::is_nothrow_move_assignable_v<memory::BlockPool>,
              "Sequence rollback must not throw");
} // namespace

Generator::Generator(GeneratorConfig config, io::WeightMap weights)
    : config_(validate_config(config)),
      block_(config_.hidden_dim, config_.intermediate_dim, config_.num_heads,
             config_.total_blocks, config_.block_size),
      pool_(config_.total_blocks, config_.block_size),
      embeddings_(take(weights, "tok_embeddings.weight",
                       {config_.vocab_size, config_.hidden_dim})),
      final_norm_(take(weights, "norm.weight", {config_.hidden_dim})),
      lm_head_(config_.hidden_dim, config_.vocab_size) {
    const auto h = config_.hidden_dim;
    block_.input_norm() = layers::RMSNorm(take(weights, "block.input_norm.weight", {h}));
    block_.post_attention_norm() = layers::RMSNorm(
        take(weights, "block.post_attention_norm.weight", {h}));
    auto& attn = block_.attention();
    load_linear(weights, "block.attn.q_proj.weight", attn.query_projection());
    load_linear(weights, "block.attn.k_proj.weight", attn.key_projection());
    load_linear(weights, "block.attn.v_proj.weight", attn.value_projection());
    load_linear(weights, "block.attn.o_proj.weight", attn.output_projection());
    load_linear(weights, "block.ffn.up_proj.weight", block_.ffn_up());
    load_linear(weights, "block.ffn.down_proj.weight", block_.ffn_down());
    load_linear(weights, "lm_head.weight", lm_head_);
    if (!weights.empty())
        throw std::invalid_argument("Unexpected weight: " + weights.begin()->first);
}

Tensor Generator::embed(const int* ids, std::size_t count) const {
    const auto h = config_.hidden_dim;
    checked_elements({count, h});
    Tensor input({count, h});
    for (std::size_t row = 0; row < count; ++row)
        std::copy_n(embeddings_.data() + static_cast<std::size_t>(ids[row]) * h,
                    h, input.data() + row * h);
    return input;
}

Tensor Generator::forward(const Tensor& input) {
    if (retry_attention_) {
        // A failed initial cache allocation can leave only one KV tensor alive.
        // Build first (may throw), then transfer learned projections noexcept.
        layers::PagedAttention clean(config_.hidden_dim, config_.num_heads,
                                     config_.total_blocks, config_.block_size);
        auto& old = block_.attention();
        clean.query_projection() = std::move(old.query_projection());
        clean.key_projection() = std::move(old.key_projection());
        clean.value_projection() = std::move(old.value_projection());
        clean.output_projection() = std::move(old.output_projection());
        old = std::move(clean);
    }
    retry_attention_ = true;
    Tensor output = block_.forward(input, 0, pages_, pool_);
    retry_attention_ = false;
    return output;
}

std::vector<int> Generator::generate(const std::vector<int>& prompt,
                                     int max_new_tokens) {
    if (prompt.empty() || max_new_tokens < 0)
        throw std::invalid_argument("Generator: nonempty prompt and nonnegative max required");
    for (int id : prompt)
        if (id < 0 || static_cast<std::size_t>(id) >= config_.vocab_size)
            throw std::invalid_argument("Generator: prompt token outside vocabulary");
    const auto extra = static_cast<std::size_t>(max_new_tokens);
    if (prompt.size() > static_cast<std::size_t>(INT_MAX) - extra)
        throw std::overflow_error("Generator: prompt plus max exceeds INT_MAX");
    if (max_new_tokens == 0 || prompt.back() == config_.eos_token_id) return prompt;
    const auto required = prompt.size() + extra;
    const auto block_size = static_cast<std::size_t>(config_.block_size);
    const auto blocks = required / block_size + (required % block_size != 0);
    if (blocks > static_cast<std::size_t>(config_.total_blocks))
        throw std::length_error("Generator: prompt plus max exceeds KV capacity");
    checked_elements({prompt.size(), config_.hidden_dim});
    checked_elements({prompt.size(), config_.intermediate_dim});
    std::vector<int> result;
    result.reserve(required);
    result.insert(result.end(), prompt.begin(), prompt.end());
    SequenceGuard guard(pages_, pool_); // Armed before sequence allocation.
    pages_.allocate_sequence(0, pool_);
    Tensor last({1, config_.hidden_dim});
    {
        Tensor hidden = forward(embed(prompt.data(), prompt.size()));
        std::copy_n(hidden.data() + (prompt.size() - 1) * config_.hidden_dim,
                    config_.hidden_dim, last.data());
    } // Retain only the last prefill row.
    for (int step = 0; step < max_new_tokens; ++step) {
        Tensor normalized = final_norm_.forward(last);
        Tensor logits = lm_head_.forward(normalized);
        const int next = sampler_.sample(logits);
        result.push_back(next);
        if (next == config_.eos_token_id || step + 1 == max_new_tokens) break;
        // Attention appends/caches this row. Never manually append the token
        // and never reuse the previous hidden state as the next embedding.
        last = forward(embed(&next, 1));
    }
    return result; // Guard restores allocator state, including on exceptions.
}
} // namespace stratakv::generation
