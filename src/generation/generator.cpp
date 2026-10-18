#include "stratakv/generation/generator.h"
#if defined(STRATAKV_HAS_CUDA)
#include "stratakv/cuda/gemm_cuda.h"
#include <cuda_runtime.h>
#endif
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
    const std::size_t kv_heads =
        c.num_kv_heads == 0 ? c.num_heads : c.num_kv_heads;
    if (c.hidden_dim == 0 || c.intermediate_dim == 0 || c.num_heads == 0 ||
        c.num_layers == 0 || kv_heads == 0 || c.num_heads % kv_heads != 0 ||
        c.vocab_size == 0 || c.vocab_size > static_cast<std::size_t>(INT_MAX) ||
        c.total_blocks <= 0 || c.block_size <= 0 ||
        c.hidden_dim % c.num_heads != 0 || c.eos_token_id < -1 ||
        (c.eos_token_id >= 0 &&
         static_cast<std::size_t>(c.eos_token_id) >= c.vocab_size))
        throw std::invalid_argument("Generator: invalid configuration");
    checked_elements({c.hidden_dim, c.hidden_dim});
    checked_elements({c.hidden_dim, c.intermediate_dim});
    checked_elements({c.vocab_size, c.hidden_dim});
    checked_elements({c.num_layers, c.hidden_dim, c.hidden_dim});
    checked_elements({c.num_layers, c.hidden_dim, c.intermediate_dim});
    const std::size_t kv_dim =
        kv_heads * (c.hidden_dim / c.num_heads);
    checked_elements({c.num_layers, c.hidden_dim, kv_dim});
    checked_elements({c.num_layers,
                      static_cast<std::size_t>(c.total_blocks),
                      static_cast<std::size_t>(c.block_size),
                      kv_dim});
    return c;
}

Device generator_device() {
#if defined(STRATAKV_HAS_CUDA)
    return Device::GPU;
#else
    return Device::CPU;
#endif
}

std::vector<models::TransformerBlock> make_blocks(const GeneratorConfig& config,
                                                  Device device) {
    std::vector<models::TransformerBlock> blocks;
    blocks.reserve(config.num_layers);
    for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
        blocks.emplace_back(config.hidden_dim, config.intermediate_dim,
                            config.num_heads, config.num_kv_heads,
                            config.total_blocks, config.block_size, device);
    }
    return blocks;
}

memory::BlockPool make_pool(const GeneratorConfig& config, Device device) {
    const int kv_heads = static_cast<int>(
        config.num_kv_heads == 0 ? config.num_heads : config.num_kv_heads);
    const int head_dim = static_cast<int>(config.hidden_dim / config.num_heads);
    if (device == Device::GPU)
        return memory::BlockPool(config.total_blocks, config.block_size,
                                 kv_heads, head_dim, device);
    return memory::BlockPool(config.total_blocks, config.block_size);
}

bool has_weight(const io::WeightMap& weights, const std::string& key) {
    return weights.find(key) != weights.end();
}

Tensor take(io::WeightMap& weights, const std::string& key,
            std::initializer_list<std::size_t> expected) {
    auto it = weights.find(key); // Never operator[]: Tensor has no default ctor.
    if (it == weights.end())
        throw std::invalid_argument("Missing weight: " + key);
    const Tensor& source = it->second;
    if (source.shape() != std::vector<std::size_t>(expected) ||
        source.num_elements() != checked_elements(expected) || !source.raw_data())
        throw std::invalid_argument("Wrong shape/storage: " + key);
    for (std::size_t i = 0; i < source.num_elements(); ++i)
        if (!std::isfinite(source[i]))
            throw std::invalid_argument("Non-finite weight: " + key);
    Tensor result = std::move(it->second);
    weights.erase(it);
    return result;
}

void load_linear(io::WeightMap& weights, const std::string& key,
                 layers::Linear& layer) {
    const auto in = layer.input_dim(), out = layer.output_dim();
    Tensor disk = take(weights, key, {out, in});
    Tensor dest({in, out}, disk.dtype());
    for (std::size_t o = 0; o < out; ++o)
        for (std::size_t i = 0; i < in; ++i)
            dest[i * out + o] = disk[o * in + i];
    layer.set_weight(std::move(dest));
    if (Tensor* bias = layer.bias())
        std::fill_n(bias->data(), bias->num_elements(), 0.0f);
}

Tensor take_embedding(io::WeightMap& weights, const GeneratorConfig& config) {
    const char* key = has_weight(weights, "model.embed_tokens.weight")
        ? "model.embed_tokens.weight"
        : "tok_embeddings.weight";
    return take(weights, key, {config.vocab_size, config.hidden_dim});
}

Tensor take_final_norm(io::WeightMap& weights, const GeneratorConfig& config) {
    const char* key = has_weight(weights, "model.norm.weight")
        ? "model.norm.weight"
        : "norm.weight";
    return take(weights, key, {config.hidden_dim});
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
      blocks_(make_blocks(config_, generator_device())),
      pool_(make_pool(config_, generator_device())),
      sampler_(config_.sampler),
      embeddings_(take_embedding(weights, config_)),
      final_norm_(take_final_norm(weights, config_)),
      lm_head_(config_.hidden_dim, config_.vocab_size) {
    const auto h = config_.hidden_dim;
    const bool llama_schema = has_weight(
        weights, "model.layers.0.input_layernorm.weight");

    if (llama_schema) {
        for (std::size_t layer_index = 0;
             layer_index < config_.num_layers; ++layer_index) {
            const std::string prefix = "model.layers." +
                std::to_string(layer_index) + ".";
            auto& block = blocks_.at(layer_index);
            block.input_norm() = layers::RMSNorm(
                take(weights, prefix + "input_layernorm.weight", {h}));
            block.post_attention_norm() = layers::RMSNorm(
                take(weights, prefix + "post_attention_layernorm.weight", {h}));

            auto& attn = block.attention();
            load_linear(weights, prefix + "self_attn.q_proj.weight",
                        attn.query_projection());
            load_linear(weights, prefix + "self_attn.k_proj.weight",
                        attn.key_projection());
            load_linear(weights, prefix + "self_attn.v_proj.weight",
                        attn.value_projection());
            load_linear(weights, prefix + "self_attn.o_proj.weight",
                        attn.output_projection());

            load_linear(weights, prefix + "mlp.gate_proj.weight",
                        block.gate_projection());
            load_linear(weights, prefix + "mlp.up_proj.weight",
                        block.up_projection());
            load_linear(weights, prefix + "mlp.down_proj.weight",
                        block.down_projection());
        }
    } else {
        // Accept the Phase 5 one-layer schema as a compatibility path. New
        // Llama checkpoints always take the branch above; this path also
        // makes the schema migration non-breaking for existing local models.
        if (config_.num_layers != 1) {
            throw std::invalid_argument(
                "Generator: Llama layer weights are missing");
        }
        auto& block = blocks_.front();
        block.input_norm() = layers::RMSNorm(
            take(weights, "block.input_norm.weight", {h}));
        block.post_attention_norm() = layers::RMSNorm(
            take(weights, "block.post_attention_norm.weight", {h}));
        auto& attn = block.attention();
        load_linear(weights, "block.attn.q_proj.weight",
                    attn.query_projection());
        load_linear(weights, "block.attn.k_proj.weight",
                    attn.key_projection());
        load_linear(weights, "block.attn.v_proj.weight",
                    attn.value_projection());
        load_linear(weights, "block.attn.o_proj.weight",
                    attn.output_projection());
        load_linear(weights, "block.ffn.up_proj.weight",
                    block.gate_projection());
        for (std::size_t i = 0;
             i < block.gate_projection().weight().num_elements(); ++i)
            block.up_projection().weight()[i] =
                block.gate_projection().weight()[i];
        load_linear(weights, "block.ffn.down_proj.weight",
                    block.down_projection());
    }
    load_linear(weights, "lm_head.weight", lm_head_);

#if defined(STRATAKV_HAS_CUDA)
    device_ = Device::GPU;
    embeddings_ = embeddings_.to(Device::GPU);
    final_norm_.to(Device::GPU);
    lm_head_.to(Device::GPU);
    for (auto& block : blocks_)
        block.to(Device::GPU);
    gpu_embedding_input_.emplace(
        std::vector<std::size_t>{1, config_.hidden_dim}, DataType::FP32,
        Device::GPU);
    gpu_final_norm_.emplace(
        std::vector<std::size_t>{1, config_.hidden_dim}, DataType::FP32,
        Device::GPU);
    gpu_logits_.emplace(
        std::vector<std::size_t>{1, config_.vocab_size}, DataType::FP32,
        Device::GPU);
    host_logits_.emplace(
        std::vector<std::size_t>{1, config_.vocab_size}, DataType::FP32,
        Device::CPU);
#endif
    if (!weights.empty())
        throw std::invalid_argument("Unexpected weight: " + weights.begin()->first);
}

Tensor Generator::embed(const int* ids, std::size_t count) const {
    const auto h = config_.hidden_dim;
    checked_elements({count, h});
    Tensor input({count, h});
    for (std::size_t row = 0; row < count; ++row)
        for (std::size_t col = 0; col < h; ++col)
            input[row * h + col] = embeddings_[
                static_cast<std::size_t>(ids[row]) * h + col];
    return input;
}

void Generator::embed_device(int token_id, Tensor& output) const {
#if defined(STRATAKV_HAS_CUDA)
    if (device_ != Device::GPU || output.device() != Device::GPU ||
        output.shape() != std::vector<std::size_t>{1, config_.hidden_dim}) {
        throw std::invalid_argument("Generator: invalid GPU embedding buffer");
    }
    embedding_lookup_cuda(
        static_cast<const float*>(embeddings_.raw_data()), token_id,
        static_cast<int>(config_.hidden_dim),
        static_cast<float*>(output.raw_data()));
#else
    (void)token_id;
    (void)output;
    throw std::runtime_error("Generator: GPU embedding requires CUDA support");
#endif
}

Tensor Generator::forward(const Tensor& input) {
    Tensor output = blocks_.front().forward(input, 0, pages_, pool_, true);
    for (std::size_t layer = 1; layer < blocks_.size(); ++layer) {
        output = blocks_[layer].forward(output, 0, pages_, pool_, false);
    }
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
    Tensor last({1, config_.hidden_dim}, DataType::FP32, device_);
    if (device_ == Device::GPU) {
#if defined(STRATAKV_HAS_CUDA)
        for (int token : prompt) {
            embed_device(token, *gpu_embedding_input_);
            last = forward(*gpu_embedding_input_);
        }
#endif
    } else {
        Tensor hidden = forward(embed(prompt.data(), prompt.size()));
        std::copy_n(hidden.data() + (prompt.size() - 1) * config_.hidden_dim,
                    config_.hidden_dim, last.data());
    }
    for (int step = 0; step < max_new_tokens; ++step) {
        int next = 0;
        if (device_ == Device::GPU) {
#if defined(STRATAKV_HAS_CUDA)
            rmsnorm_cuda(
                static_cast<const float*>(last.raw_data()),
                static_cast<float*>(gpu_final_norm_->raw_data()),
                static_cast<const float*>(final_norm_.gamma().raw_data()),
                static_cast<int>(config_.hidden_dim), final_norm_.epsilon());
            matmul_cuda_tiled(*gpu_final_norm_, lm_head_.weight(),
                              *gpu_logits_);
            const auto copy_status = cudaMemcpy(
                host_logits_->raw_data(), gpu_logits_->raw_data(),
                config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
            if (copy_status != cudaSuccess)
                throw std::runtime_error(
                    std::string("Generator: logits copy failed: ") +
                    cudaGetErrorString(copy_status));
            next = sampler_.sample(*host_logits_);
#endif
        } else {
            Tensor normalized = final_norm_.forward(last);
            Tensor logits = lm_head_.forward(normalized);
            next = sampler_.sample(logits);
        }
        result.push_back(next);
        if (next == config_.eos_token_id || step + 1 == max_new_tokens) break;
        // Attention appends/caches this row. Never manually append the token
        // and never reuse the previous hidden state as the next embedding.
        if (device_ == Device::GPU) {
#if defined(STRATAKV_HAS_CUDA)
            embed_device(next, *gpu_embedding_input_);
            last = forward(*gpu_embedding_input_);
#endif
        } else {
            last = forward(embed(&next, 1));
        }
    }
    return result; // Guard restores allocator state, including on exceptions.
}
} // namespace stratakv::generation
