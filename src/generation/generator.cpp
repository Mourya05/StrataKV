#include "stratakv/generation/generator.h"
#if defined(STRATAKV_HAS_CUDA)
#include "stratakv/cuda/gemm_cuda.h"
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <thread>
#include <utility>
#include <cstdlib>
#if defined(STRATAKV_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace stratakv::generation {
namespace {
constexpr std::size_t kParallelThreshold = 4u * 1024u * 1024u;

bool parity_enabled() {
    const char* value = std::getenv("STRATAKV_PARITY");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

template <typename Function>
void parallel_for(std::size_t count, Function function) {
    const unsigned workers = std::thread::hardware_concurrency();
    if (count < kParallelThreshold || workers < 2) {
        function(0, count);
        return;
    }
    const std::size_t thread_count =
        (std::min)(std::size_t{8}, static_cast<std::size_t>(workers));
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    const std::size_t chunk = (count + thread_count - 1) / thread_count;
    for (std::size_t begin = 0; begin < count; begin += chunk) {
        const std::size_t end = (std::min)(count, begin + chunk);
        threads.emplace_back([=, &function] { function(begin, end); });
    }
    for (auto& thread : threads)
        thread.join();
}

#if defined(STRATAKV_HAS_CUDA)
std::vector<float> host_values_cpu(const Tensor& tensor) {
    std::vector<float> values(tensor.num_elements());
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = tensor[i];
    return values;
}

std::vector<float> parity_host_values(const Tensor& tensor) {
    std::vector<float> values(tensor.num_elements());
    const std::size_t bytes = tensor.num_elements() * tensor.element_size();
    std::vector<unsigned char> raw(bytes);
    if (cudaMemcpy(raw.data(), tensor.raw_data(), bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("Generator: parity device copy failed");
    if (tensor.dtype() == DataType::FP32) {
        std::memcpy(values.data(), raw.data(), bytes);
    } else {
        const auto* data = reinterpret_cast<const std::uint16_t*>(raw.data());
        for (std::size_t i = 0; i < values.size(); ++i) {
            const std::uint32_t bits =
                static_cast<std::uint32_t>(data[i]) << 16u;
            std::memcpy(&values[i], &bits, sizeof(float));
        }
    }
    return values;
}

void parity_final_report(const std::vector<float>& gpu,
                         const std::vector<float>& cpu) {
    float max_diff = 0.0f;
    std::size_t gpu_argmax = 0;
    std::size_t cpu_argmax = 0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        max_diff = (std::max)(max_diff, std::fabs(gpu[i] - cpu[i]));
        if (gpu[i] > gpu[gpu_argmax]) gpu_argmax = i;
        if (cpu[i] > cpu[cpu_argmax]) cpu_argmax = i;
    }
    std::fprintf(stderr,
                 "[PARITY FINAL] lm_head logits: max_diff = %.9g, "
                 "gpu_argmax = %zu, cpu_argmax = %zu\n",
                 max_diff, gpu_argmax, cpu_argmax);
}

void parity_final_step(int position, const Tensor& gpu_hidden,
                       const Tensor& cpu_hidden, const layers::RMSNorm& norm,
                       const layers::Linear& lm_head,
                       Tensor& gpu_norm, Tensor& gpu_logits) {
    rmsnorm_cuda(static_cast<const float*>(gpu_hidden.raw_data()),
                 static_cast<float*>(gpu_norm.raw_data()),
                 static_cast<const float*>(norm.gamma().raw_data()),
                 static_cast<int>(gpu_hidden.shape()[1]), norm.epsilon());
    matmul_cuda_tiled(gpu_norm, lm_head.weight(), gpu_logits);

    const auto gpu_norm_values = parity_host_values(gpu_norm);
    const auto cpu_hidden_values = host_values_cpu(cpu_hidden);
    const auto gamma_values = host_values_cpu(norm.gamma().to(Device::CPU));
    const float sum = std::accumulate(
        cpu_hidden_values.begin(), cpu_hidden_values.end(), 0.0f,
        [](float total, float value) { return total + value * value; });
    const float scale = 1.0f /
        std::sqrt(sum / cpu_hidden_values.size() + norm.epsilon());
    std::vector<float> cpu_norm(cpu_hidden_values.size());
    for (std::size_t i = 0; i < cpu_norm.size(); ++i)
        cpu_norm[i] = cpu_hidden_values[i] * scale * gamma_values[i];
    float norm_diff = 0.0f;
    for (std::size_t i = 0; i < cpu_norm.size(); ++i)
        norm_diff = (std::max)(norm_diff,
                               std::fabs(gpu_norm_values[i] - cpu_norm[i]));
    std::fprintf(stderr, "[PARITY pos=%d] final_norm max_diff = %.9g\n",
                 position, norm_diff);

    const auto weight_values = host_values_cpu(lm_head.weight().to(Device::CPU));
    std::vector<float> cpu_logits(lm_head.output_dim(), 0.0f);
    for (std::size_t col = 0; col < lm_head.output_dim(); ++col)
        for (std::size_t row = 0; row < lm_head.input_dim(); ++row)
            cpu_logits[col] += cpu_norm[row] *
                weight_values[row * lm_head.output_dim() + col];
    parity_final_report(parity_host_values(gpu_logits), cpu_logits);
}
#endif

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

std::vector<memory::BlockPool> make_pools(const GeneratorConfig& config,
                                          Device device) {
    std::vector<memory::BlockPool> pools;
    pools.reserve(config.num_layers);
    for (std::size_t i = 0; i < config.num_layers; ++i)
        pools.push_back(make_pool(config, device));
    return pools;
}

std::vector<memory::PageTable> make_pages(const GeneratorConfig& config) {
    return std::vector<memory::PageTable>(config.num_layers);
}

#if defined(STRATAKV_HAS_CUDA)
std::vector<models::TransformerBlock> make_parity_blocks(
    const GeneratorConfig& config,
    const std::vector<models::TransformerBlock>& source) {
    std::vector<models::TransformerBlock> result;
    result.reserve(config.num_layers);
    for (std::size_t i = 0; i < config.num_layers; ++i) {
        result.emplace_back(config.hidden_dim, config.intermediate_dim,
                            config.num_heads, config.num_kv_heads,
                            config.total_blocks, config.block_size,
                            Device::CPU);
        auto& dst = result.back();
        const auto& src = source[i];
        dst.input_norm() = layers::RMSNorm(src.input_norm().gamma().to(Device::CPU));
        dst.post_attention_norm() =
            layers::RMSNorm(src.post_attention_norm().gamma().to(Device::CPU));
        auto copy_linear = [](const layers::Linear& from, layers::Linear& to) {
            to.set_weight(from.weight().to(Device::CPU));
        };
        copy_linear(src.attention().query_projection(),
                    dst.attention().query_projection());
        copy_linear(src.attention().key_projection(),
                    dst.attention().key_projection());
        copy_linear(src.attention().value_projection(),
                    dst.attention().value_projection());
        copy_linear(src.attention().output_projection(),
                    dst.attention().output_projection());
        copy_linear(src.gate_projection(), dst.gate_projection());
        copy_linear(src.up_projection(), dst.up_projection());
        copy_linear(src.down_projection(), dst.down_projection());
    }
    return result;
}
#endif

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
    std::atomic<bool> finite{true};
    parallel_for(source.num_elements(), [&](std::size_t begin,
                                            std::size_t end) {
        for (std::size_t i = begin; i < end && finite.load(); ++i) {
            if (!std::isfinite(source[i])) {
                finite.store(false);
                break;
            }
        }
    });
    if (!finite.load())
        throw std::invalid_argument("Non-finite weight: " + key);
    Tensor result = std::move(it->second);
    weights.erase(it);
    return result;
}

void load_linear(io::WeightMap& weights, const std::string& key,
                 layers::Linear& layer, Tensor* gpu_staging = nullptr) {
    const auto in = layer.input_dim(), out = layer.output_dim();
    Tensor disk = take(weights, key, {out, in});
#if defined(STRATAKV_HAS_CUDA)
    if (gpu_staging != nullptr) {
        if (disk.dtype() != DataType::FP32 ||
            gpu_staging->dtype() != DataType::FP32 ||
            gpu_staging->device() != Device::GPU ||
            gpu_staging->num_elements() < disk.num_elements()) {
            throw std::invalid_argument(
                "Generator: invalid GPU linear staging buffer");
        }
        const auto status = cudaMemcpy(
            gpu_staging->raw_data(), disk.raw_data(),
            disk.num_elements() * sizeof(float), cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
            throw std::runtime_error(
                std::string("Generator: linear staging upload failed: ") +
                cudaGetErrorString(status));
        Tensor destination({in, out}, DataType::BF16, Device::GPU);
        transpose_fp32_to_bf16_cuda(
            static_cast<const float*>(gpu_staging->raw_data()),
            static_cast<std::uint16_t*>(destination.raw_data()),
            static_cast<int>(out), static_cast<int>(in));
        layer.set_weight(std::move(destination));
        if (Tensor* bias = layer.bias())
            std::fill_n(bias->data(), bias->num_elements(), 0.0f);
        return;
    }
#endif
    Tensor dest = io::load_linear_bf16_transposed(disk, out, in);
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
    SequenceGuard(std::vector<memory::PageTable>& pages,
                  std::vector<memory::BlockPool>& pools)
        : pages_(pages), pools_(pools), saved_pages_(pages), saved_pools_(pools) {}
    SequenceGuard(const SequenceGuard&) = delete;
    SequenceGuard& operator=(const SequenceGuard&) = delete;
    ~SequenceGuard() noexcept {
        try {
            for (std::size_t i = 0; i < pages_.size(); ++i)
                if (pages_[i].has_sequence(0))
                    pages_[i].free_sequence(0, pools_[i]);
        } catch (...) { /* Snapshot rollback also handles cleanup failure. */ }
        pages_ = std::move(saved_pages_);
        pools_ = std::move(saved_pools_);
    }
private:
    std::vector<memory::PageTable>& pages_;
    std::vector<memory::BlockPool>& pools_;
    std::vector<memory::PageTable> saved_pages_;
    std::vector<memory::BlockPool> saved_pools_;
};
static_assert(std::is_nothrow_move_assignable_v<memory::PageTable> &&
              std::is_nothrow_move_assignable_v<memory::BlockPool>,
              "Sequence rollback must not throw");
} // namespace

Generator::Generator(GeneratorConfig config, io::WeightMap weights)
    : config_(validate_config(config)),
      blocks_(make_blocks(config_, generator_device())),
      pools_(make_pools(config_, generator_device())),
      pages_(make_pages(config_)),
      sampler_(config_.sampler),
      embeddings_(take_embedding(weights, config_)),
      final_norm_(take_final_norm(weights, config_)),
      lm_head_(config_.hidden_dim, config_.vocab_size) {
    const auto init_start = std::chrono::steady_clock::now();
    const auto h = config_.hidden_dim;
#if defined(STRATAKV_HAS_CUDA)
    std::optional<Tensor> gpu_staging;
    {
        const std::size_t largest_linear = (std::max)(
            config_.vocab_size * h, config_.intermediate_dim * h);
        gpu_staging.emplace(
            std::vector<std::size_t>{largest_linear},
            DataType::FP32, Device::GPU);
    }
    Tensor* staging = gpu_staging ? &*gpu_staging : nullptr;
#else
    Tensor* staging = nullptr;
#endif
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
                        attn.query_projection(), staging);
            load_linear(weights, prefix + "self_attn.k_proj.weight",
                        attn.key_projection(), staging);
            load_linear(weights, prefix + "self_attn.v_proj.weight",
                        attn.value_projection(), staging);
            load_linear(weights, prefix + "self_attn.o_proj.weight",
                        attn.output_projection(), staging);

            load_linear(weights, prefix + "mlp.gate_proj.weight",
                        block.gate_projection(), staging);
            load_linear(weights, prefix + "mlp.up_proj.weight",
                        block.up_projection(), staging);
            load_linear(weights, prefix + "mlp.down_proj.weight",
                        block.down_projection(), staging);
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
                    attn.query_projection(), staging);
        load_linear(weights, "block.attn.k_proj.weight",
                    attn.key_projection(), staging);
        load_linear(weights, "block.attn.v_proj.weight",
                    attn.value_projection(), staging);
        load_linear(weights, "block.attn.o_proj.weight",
                    attn.output_projection(), staging);
        load_linear(weights, "block.ffn.up_proj.weight",
                    block.gate_projection(), staging);
        for (std::size_t i = 0;
             i < block.gate_projection().weight().num_elements(); ++i)
            block.up_projection().weight()[i] =
                block.gate_projection().weight()[i];
        load_linear(weights, "block.ffn.down_proj.weight",
                    block.down_projection(), staging);
    }
    load_linear(weights, "lm_head.weight", lm_head_, staging);
#if defined(STRATAKV_HAS_CUDA)
    if (gpu_staging) {
        cudaDeviceSynchronize();
        gpu_staging.reset();
    }
#endif

#if defined(STRATAKV_HAS_CUDA)
    device_ = Device::GPU;
    if (embeddings_.dtype() != DataType::FP32) {
        Tensor embedding_fp32(embeddings_.shape(), DataType::FP32);
        parallel_for(embeddings_.num_elements(),
                     [&](std::size_t begin, std::size_t end) {
                         for (std::size_t i = begin; i < end; ++i)
                             embedding_fp32[i] = embeddings_[i];
                     });
        embeddings_ = std::move(embedding_fp32);
    }
    embeddings_ = embeddings_.to(Device::GPU);
    final_norm_.to(Device::GPU);
    lm_head_.to(Device::GPU);
    for (auto& block : blocks_)
        block.to(Device::GPU);
    if (parity_enabled()) {
        blocks_.front().set_parity_diagnostic(true);
        parity_blocks_ = make_parity_blocks(config_, blocks_);
        parity_pools_ = make_pools(config_, Device::CPU);
        parity_pages_ = make_pages(config_);
        parity_last_cpu_.emplace(
            std::vector<std::size_t>{1, config_.hidden_dim},
            DataType::FP32, Device::CPU);
    }
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
    initialization_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - init_start).count();
}

std::size_t Generator::free_block_count() const noexcept {
    std::size_t total = 0;
    for (const auto& pool : pools_)
        total += pool.free_count();
    return total;
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
        output.shape() != std::vector<std::size_t>{1, config_.hidden_dim} ||
        embeddings_.device() != Device::GPU ||
        embeddings_.dtype() != DataType::FP32 ||
        embeddings_.shape() !=
            std::vector<std::size_t>{config_.vocab_size, config_.hidden_dim} ||
        token_id < 0 ||
        static_cast<std::size_t>(token_id) >= config_.vocab_size) {
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
    Tensor output = input.to(input.device());
#if defined(STRATAKV_HAS_CUDA)
    std::vector<std::vector<float>> gpu_layer_values;
    if (!parity_blocks_.empty())
        gpu_layer_values.reserve(blocks_.size());
#endif
    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        if (device_ == Device::GPU) {
#if defined(STRATAKV_HAS_CUDA)
            blocks_[layer].forward_into(
                output, output, 0, pages_[layer], pools_[layer], true);
#endif
        } else {
            output = blocks_[layer].forward(
                output, 0, pages_[layer], pools_[layer], true);
        }
#if defined(STRATAKV_HAS_CUDA)
        if (!parity_blocks_.empty())
            gpu_layer_values.push_back(parity_host_values(output));
#endif
    }
#if defined(STRATAKV_HAS_CUDA)
    if (!parity_blocks_.empty()) {
        Tensor cpu_input = input.to(Device::CPU);
        Tensor cpu_output = std::move(cpu_input);
        for (std::size_t layer = 0; layer < parity_blocks_.size(); ++layer) {
            cpu_output = parity_blocks_[layer].forward(
                cpu_output, 0, parity_pages_[layer], parity_pools_[layer],
                true);
            const std::vector<float>& gpu_values = gpu_layer_values[layer];
            float max_diff = 0.0f;
            for (std::size_t i = 0; i < gpu_values.size(); ++i)
                max_diff = (std::max)(max_diff,
                    std::fabs(gpu_values[i] - cpu_output[i]));
            std::fprintf(stderr, "[PARITY pos=%d layer=%zu] max_diff = %.9g\n",
                         parity_pages_[layer].get_state(0).sequence_length - 1,
                         layer, max_diff);
        }
        *parity_last_cpu_ = std::move(cpu_output);
    }
#endif
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
    const auto generation_start = std::chrono::steady_clock::now();
    std::vector<int> result;
    result.reserve(required);
    result.insert(result.end(), prompt.begin(), prompt.end());
    SequenceGuard guard(pages_, pools_); // Armed before sequence allocation.
    for (std::size_t layer = 0; layer < blocks_.size(); ++layer)
        pages_[layer].allocate_sequence(0, pools_[layer]);
#if defined(STRATAKV_HAS_CUDA)
    if (parity_enabled()) {
        for (std::size_t layer = 0; layer < parity_pages_.size(); ++layer) {
            if (!parity_pages_[layer].has_sequence(0))
                parity_pages_[layer].allocate_sequence(
                    0, parity_pools_[layer]);
        }
    }
#endif
    Tensor last({1, config_.hidden_dim}, DataType::FP32, device_);
    if (device_ == Device::GPU) {
#if defined(STRATAKV_HAS_CUDA)
        const bool parity = parity_enabled();
        bool first_token = true;
        int prompt_position = 0;
        for (int token : prompt) {
            embed_device(token, *gpu_embedding_input_);
            if (parity && first_token) {
                const auto gpu = parity_host_values(*gpu_embedding_input_);
                const auto embedding = parity_host_values(embeddings_);
                std::vector<float> cpu(config_.hidden_dim);
                for (std::size_t i = 0; i < config_.hidden_dim; ++i)
                    cpu[i] = embedding[static_cast<std::size_t>(token) *
                                       config_.hidden_dim + i];
                float max_diff = 0.0f;
                for (std::size_t i = 0; i < cpu.size(); ++i)
                    max_diff = (std::max)(max_diff,
                                          std::fabs(gpu[i] - cpu[i]));
                std::fprintf(stderr,
                             "[PARITY L0] embedding: max_diff = %.9g, "
                             "gpu_sample[0..2] = [%.9g, %.9g, %.9g], "
                             "cpu_sample[0..2] = [%.9g, %.9g, %.9g]\n",
                             max_diff, gpu[0], gpu[1], gpu[2],
                             cpu[0], cpu[1], cpu[2]);
            }
            last = forward(*gpu_embedding_input_);
            if (parity && prompt_position < 2)
                parity_final_step(prompt_position, last, *parity_last_cpu_,
                                  final_norm_, lm_head_, *gpu_final_norm_,
                                  *gpu_logits_);
            first_token = false;
            ++prompt_position;
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
            if (lm_head_.weight().device() != Device::GPU ||
                lm_head_.weight().dtype() != DataType::BF16 ||
                lm_head_.weight().shape() !=
                    std::vector<std::size_t>{
                        config_.hidden_dim, config_.vocab_size} ||
                gpu_logits_->shape() !=
                    std::vector<std::size_t>{1, config_.vocab_size}) {
                throw std::runtime_error(
                    "Generator: invalid GPU LM head boundary");
            }
            matmul_cuda_tiled(*gpu_final_norm_, lm_head_.weight(),
                              *gpu_logits_);
            const auto copy_status = cudaMemcpy(
                host_logits_->raw_data(), gpu_logits_->raw_data(),
                config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
            if (copy_status != cudaSuccess)
                throw std::runtime_error(
                    std::string("Generator: logits copy failed: ") +
                    cudaGetErrorString(copy_status));
            for (std::size_t i = 0; i < host_logits_->num_elements(); ++i) {
                const float value = (*host_logits_)[i];
                if (std::isnan(value) || std::isinf(value)) {
                    std::cerr
                        << "CRITICAL NUMERICAL COLLAPSE: NaN/Inf detected in "
                           "logits! Safetensors dtype mismatch (FP16 weights "
                           "fed to BF16 kernel).\n";
                    std::exit(1);
                }
            }
            if (host_logits_->shape() !=
                std::vector<std::size_t>{1, config_.vocab_size} ||
                host_logits_->num_elements() != config_.vocab_size) {
                throw std::runtime_error(
                    "Generator: LM head produced invalid vocabulary shape");
            }
            next = sampler_.sample(*host_logits_);
#if defined(STRATAKV_HAS_CUDA)
            if (parity_enabled() &&
                step == 0) {
                if (!parity_last_cpu_)
                    throw std::runtime_error(
                        "Generator: missing CPU parity hidden state");
                const auto hidden = parity_last_cpu_->to(Device::CPU);
                const auto gamma = final_norm_.gamma().to(Device::CPU);
                const auto normalized = host_values_cpu(hidden);
                const auto gamma_values = host_values_cpu(gamma);
                const float sum = std::accumulate(
                    normalized.begin(), normalized.end(), 0.0f,
                    [](float total, float value) {
                        return total + value * value;
                    });
                const float scale = 1.0f /
                    std::sqrt(sum / normalized.size() +
                              final_norm_.epsilon());
                std::vector<float> norm(normalized.size());
                for (std::size_t i = 0; i < norm.size(); ++i)
                    norm[i] = normalized[i] * scale * gamma_values[i];
                const auto weight = lm_head_.weight().to(Device::CPU);
                const auto weight_values = host_values_cpu(weight);
                std::vector<float> cpu(config_.vocab_size, 0.0f);
                for (std::size_t col = 0; col < config_.vocab_size; ++col)
                    for (std::size_t row = 0; row < config_.hidden_dim; ++row)
                        cpu[col] += norm[row] *
                            weight_values[row * config_.vocab_size + col];
                parity_final_report(parity_host_values(*gpu_logits_), cpu);
            }
#endif
#endif
        } else {
            Tensor normalized = final_norm_.forward(last);
            Tensor logits = lm_head_.forward(normalized);
            for (std::size_t i = 0; i < logits.num_elements(); ++i) {
                const float value = logits[i];
                if (std::isnan(value) || std::isinf(value)) {
                    std::cerr
                        << "CRITICAL NUMERICAL COLLAPSE: NaN/Inf detected in "
                           "logits! Safetensors dtype mismatch (FP16 weights "
                           "fed to BF16 kernel).\n";
                    std::exit(1);
                }
            }
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
    last_generation_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - generation_start).count();
    return result; // Guard restores allocator state, including on exceptions.
}
} // namespace stratakv::generation
