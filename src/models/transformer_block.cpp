#include "stratakv/models/transformer_block.h"
#include "stratakv/ops.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(STRATAKV_HAS_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace stratakv {
namespace models {

namespace {

#if defined(STRATAKV_HAS_CUDA)
void copy_device(const void* source, void* target, std::size_t bytes,
                 const char* operation) {
    const cudaError_t status = cudaMemcpy(
        target, source, bytes, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess)
        throw std::runtime_error(
            std::string("TransformerBlock: ") + operation + " failed: " +
            cudaGetErrorString(status));
}

std::vector<float> host_values(const Tensor& tensor) {
    if (!tensor.is_contiguous() && tensor.device() == Device::GPU)
        return host_values(tensor.to(Device::CPU));
    std::vector<float> values(tensor.num_elements());
    if (tensor.device() == Device::CPU) {
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = tensor[i];
        return values;
    }
    const std::size_t bytes = tensor.num_elements() * tensor.element_size();
    std::vector<unsigned char> raw(bytes);
    const cudaError_t status = cudaMemcpy(
        raw.data(), tensor.raw_data(), bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("TransformerBlock: parity copy failed: ") +
                                 cudaGetErrorString(status));
    if (tensor.dtype() == DataType::FP32) {
        std::memcpy(values.data(), raw.data(), bytes);
    } else {
        const auto* bf16 = reinterpret_cast<const std::uint16_t*>(raw.data());
        for (std::size_t i = 0; i < values.size(); ++i) {
            const std::uint32_t bits =
                static_cast<std::uint32_t>(bf16[i]) << 16u;
            std::memcpy(&values[i], &bits, sizeof(float));
        }
    }
    return values;
}

void parity_report(const char* name, const std::vector<float>& gpu,
                   const std::vector<float>& cpu) {
    float max_diff = 0.0f;
    const std::size_t count = (std::min)(gpu.size(), cpu.size());
    for (std::size_t i = 0; i < count; ++i)
        max_diff = (std::max)(max_diff, std::fabs(gpu[i] - cpu[i]));
    std::fprintf(stderr,
                 "[PARITY L0] %s: max_diff = %.9g, gpu_sample[0..2] = "
                 "[%.9g, %.9g, %.9g], cpu_sample[0..2] = [%.9g, %.9g, %.9g]\n",
                 name, max_diff,
                 gpu.size() > 0 ? gpu[0] : 0.0f,
                 gpu.size() > 1 ? gpu[1] : 0.0f,
                 gpu.size() > 2 ? gpu[2] : 0.0f,
                 cpu.size() > 0 ? cpu[0] : 0.0f,
                 cpu.size() > 1 ? cpu[1] : 0.0f,
                 cpu.size() > 2 ? cpu[2] : 0.0f);
}

std::vector<float> cpu_rmsnorm(const std::vector<float>& input,
                               const std::vector<float>& weight,
                               float epsilon) {
    float sum = 0.0f;
    for (float value : input) sum += value * value;
    const float scale = 1.0f / std::sqrt(sum / input.size() + epsilon);
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
        output[i] = input[i] * scale * weight[i];
    return output;
}

std::vector<float> cpu_matmul(const std::vector<float>& input,
                              std::size_t input_dim, const Tensor& weight) {
    const std::vector<float> w = host_values(weight);
    const std::size_t output_dim = weight.shape()[1];
    std::vector<float> output(output_dim, 0.0f);
    for (std::size_t col = 0; col < output_dim; ++col)
        for (std::size_t row = 0; row < input_dim; ++row)
            output[col] += input[row] * w[row * output_dim + col];
    return output;
}

void cpu_rope_split(std::vector<float>& values, const std::vector<float>& cos,
                    const std::vector<float>& sin, int heads, int head_dim) {
    const int half = head_dim / 2;
    for (int head = 0; head < heads; ++head) {
        const std::size_t offset = static_cast<std::size_t>(head) * head_dim;
        for (int i = 0; i < half; ++i) {
            const float x1 = values[offset + i];
            const float x2 = values[offset + half + i];
            values[offset + i] = x1 * cos[i] - x2 * sin[i];
            values[offset + half + i] = x2 * cos[i] + x1 * sin[i];
        }
    }
}

void cpu_swiglu(std::vector<float>& values, int intermediate_dim) {
    for (int i = 0; i < intermediate_dim; ++i) {
        const float gate = values[i];
        const float sigmoid = gate >= 0.0f
            ? 1.0f / (1.0f + std::exp(-gate))
            : std::exp(gate) / (1.0f + std::exp(gate));
        values[i] = gate * sigmoid * values[intermediate_dim + i];
    }
    values.resize(static_cast<std::size_t>(intermediate_dim));
}

std::vector<float> cpu_attention(const std::vector<float>& query,
                                 const memory::BlockPool& pool,
                                 const memory::SequenceState& state,
                                 int num_heads, int num_kv_heads,
                                 int head_dim, float scale) {
    const std::size_t cache_count =
        static_cast<std::size_t>(pool.total_blocks()) *
        static_cast<std::size_t>(pool.block_size()) *
        static_cast<std::size_t>(num_kv_heads) *
        static_cast<std::size_t>(head_dim);
    std::vector<float> keys(cache_count), values(cache_count);
    cudaMemcpy(keys.data(), pool.key_cache(), cache_count * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(values.data(), pool.value_cache(), cache_count * sizeof(float),
               cudaMemcpyDeviceToHost);
    std::vector<float> output(query.size(), 0.0f);
    const int q_per_kv = num_heads / num_kv_heads;
    const std::size_t token_stride =
        static_cast<std::size_t>(num_kv_heads) * head_dim;
    for (int q_head = 0; q_head < num_heads; ++q_head) {
        const int kv_head = q_head / q_per_kv;
        std::vector<float> logits(state.sequence_length);
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < state.sequence_length; ++token) {
            const int physical = state.physical_blocks[token / pool.block_size()];
            const std::size_t offset =
                (static_cast<std::size_t>(physical) * pool.block_size() +
                 token % pool.block_size()) * token_stride +
                static_cast<std::size_t>(kv_head) * head_dim;
            float dot = 0.0f;
            for (int dim = 0; dim < head_dim; ++dim)
                dot += query[static_cast<std::size_t>(q_head) * head_dim + dim] *
                    keys[offset + dim];
            logits[token] = dot * scale;
            max_logit = (std::max)(max_logit, logits[token]);
        }
        float total = 0.0f;
        for (float& logit : logits) {
            logit = std::exp(logit - max_logit);
            total += logit;
        }
        for (int token = 0; token < state.sequence_length; ++token) {
            const int physical = state.physical_blocks[token / pool.block_size()];
            const std::size_t offset =
                (static_cast<std::size_t>(physical) * pool.block_size() +
                 token % pool.block_size()) * token_stride +
                static_cast<std::size_t>(kv_head) * head_dim;
            const float attention_weight = logits[token] / total;
            for (int dim = 0; dim < head_dim; ++dim)
                output[static_cast<std::size_t>(q_head) * head_dim + dim] +=
                    attention_weight * values[offset + dim];
        }
    }
    return output;
}
#endif

} // namespace

TransformerBlock::TransformerBlock(std::size_t hidden_dim,
                                   std::size_t intermediate_dim,
                                   std::size_t num_heads,
                                   int total_blocks,
                                   int block_size,
                                   Device device)
    : TransformerBlock(hidden_dim, intermediate_dim, num_heads, 0,
                       total_blocks, block_size, device) {}

TransformerBlock::TransformerBlock(std::size_t hidden_dim,
                                   std::size_t intermediate_dim,
                                   std::size_t num_heads,
                                   std::size_t num_kv_heads,
                                   int total_blocks,
                                   int block_size,
                                   Device device)
    : hidden_dim_(hidden_dim)
    , norm1_(hidden_dim)
    , attention_(hidden_dim, num_heads, num_kv_heads, total_blocks, block_size,
                 device)
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

void TransformerBlock::to(Device device) {
    norm1_.to(device);
    attention_.query_projection().to(device);
    attention_.key_projection().to(device);
    attention_.value_projection().to(device);
    attention_.output_projection().to(device);
    norm2_.to(device);
    gate_proj_.to(device);
    up_proj_.to(device);
    down_proj_.to(device);
#if defined(STRATAKV_HAS_CUDA)
    if (device == Device::GPU) {
        const std::size_t kv_dim =
            attention_.num_kv_heads() * attention_.head_dim();
        normalized_input_workspace_.emplace(
            std::vector<std::size_t>{1, hidden_dim_}, DataType::FP32, device);
        query_workspace_.emplace(
            std::vector<std::size_t>{1, hidden_dim_}, DataType::FP32, device);
        key_workspace_.emplace(
            std::vector<std::size_t>{1, kv_dim}, DataType::FP32, device);
        value_workspace_.emplace(
            std::vector<std::size_t>{1, kv_dim}, DataType::FP32, device);
        cos_workspace_.emplace(
            std::vector<std::size_t>{attention_.head_dim() / 2},
            DataType::FP32, device);
        sin_workspace_.emplace(
            std::vector<std::size_t>{attention_.head_dim() / 2},
            DataType::FP32, device);
        context_workspace_.emplace(
            std::vector<std::size_t>{1, hidden_dim_}, DataType::FP32, device);
        normalized_residual_workspace_.emplace(
            std::vector<std::size_t>{1, hidden_dim_}, DataType::FP32, device);
        gate_workspace_.emplace(
            std::vector<std::size_t>{1, gate_proj_.output_dim()},
            DataType::FP32, device);
        up_workspace_.emplace(
            std::vector<std::size_t>{1, up_proj_.output_dim()},
            DataType::FP32, device);
        down_workspace_.emplace(
            std::vector<std::size_t>{1, hidden_dim_}, DataType::FP32, device);
    } else {
        normalized_input_workspace_.reset();
        query_workspace_.reset();
        key_workspace_.reset();
        value_workspace_.reset();
        cos_workspace_.reset();
        sin_workspace_.reset();
        context_workspace_.reset();
        normalized_residual_workspace_.reset();
        gate_workspace_.reset();
        up_workspace_.reset();
        down_workspace_.reset();
    }
#endif
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

#if defined(STRATAKV_HAS_CUDA)
    if (hidden_states.device() == Device::GPU) {
        Tensor result({1, hidden_dim_}, DataType::FP32, Device::GPU);
        forward_into(hidden_states, result, seq_id, page_table, block_pool,
                     append_tokens);
        return result;
    }
#else
    if (hidden_states.device() == Device::GPU) {
        throw std::runtime_error(
            "TransformerBlock::forward: GPU tensors require CUDA support");
    }
#endif

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

void TransformerBlock::forward_into(
    const Tensor& hidden_states, Tensor& output, int seq_id,
    memory::PageTable& page_table, memory::BlockPool& block_pool,
    bool append_tokens) {
    if (hidden_states.ndim() != 2 ||
        hidden_states.shape()[1] != hidden_dim_ ||
        output.device() != hidden_states.device() ||
        output.shape() != hidden_states.shape() ||
        output.dtype() != DataType::FP32) {
        throw std::invalid_argument(
            "TransformerBlock::forward_into: output shape or device mismatch");
    }
#if defined(STRATAKV_HAS_CUDA)
    if (output.device() == Device::GPU) {
        if (hidden_states.shape()[0] != 1) {
            throw std::invalid_argument(
                "TransformerBlock::forward_into: GPU decode requires one token");
        }
        if (hidden_states.dtype() != DataType::FP32) {
            throw std::runtime_error(
                "TransformerBlock::forward_into: hidden_states is not FP32");
        }
        if (norm1_.gamma().dtype() != DataType::FP32 ||
            norm2_.gamma().dtype() != DataType::FP32 ||
            norm1_.gamma().device() != Device::GPU ||
            norm2_.gamma().device() != Device::GPU) {
            throw std::runtime_error(
                "TransformerBlock::forward_into: RMSNorm weights are not on GPU");
        }
        if (attention_.query_projection().weight().device() != Device::GPU ||
            attention_.key_projection().weight().device() != Device::GPU ||
            attention_.value_projection().weight().device() != Device::GPU ||
            attention_.output_projection().weight().device() != Device::GPU ||
            gate_proj_.weight().device() != Device::GPU ||
            up_proj_.weight().device() != Device::GPU ||
            down_proj_.weight().device() != Device::GPU) {
            throw std::runtime_error(
                "TransformerBlock::forward_into: projection weights are not on GPU");
        }
        if (block_pool.device() != Device::GPU ||
            block_pool.key_cache() == nullptr ||
            block_pool.value_cache() == nullptr ||
            block_pool.num_kv_heads() !=
                static_cast<int>(attention_.num_kv_heads()) ||
            block_pool.head_dim() !=
                static_cast<int>(attention_.head_dim())) {
            throw std::invalid_argument(
                "TransformerBlock::forward_into: GPU BlockPool configuration does "
                "not match attention");
        }
        if (!page_table.has_sequence(seq_id)) {
            throw std::runtime_error(
                "TransformerBlock::forward_into: seq_id is not registered");
        }

        const memory::SequenceState& initial_state =
            page_table.get_state(seq_id);
        if (initial_state.sequence_length <= 0 && !append_tokens) {
            throw std::invalid_argument(
                "TransformerBlock::forward_into: cannot decode without context");
        }
        const int logical_position = initial_state.sequence_length;
        if (append_tokens)
            page_table.append_token(seq_id, block_pool);
        const memory::SequenceState& state = page_table.get_state(seq_id);
        const int context_len = state.sequence_length;
        const int position = append_tokens ? logical_position : context_len - 1;
        const int block_size = block_pool.block_size();
        const int head_dim = static_cast<int>(attention_.head_dim());
        const int num_heads = static_cast<int>(attention_.num_heads());
        const int num_kv_heads = static_cast<int>(attention_.num_kv_heads());
        const bool parity = parity_diagnostic_ && seq_id == 0 &&
            append_tokens && logical_position == 0;
        const std::vector<float> parity_input =
            parity ? host_values(hidden_states) : std::vector<float>();
        std::vector<float> parity_norm1;
        if (parity)
            parity_norm1 = cpu_rmsnorm(
                parity_input, host_values(norm1_.gamma()), norm1_.epsilon());

        if (!normalized_input_workspace_ || !query_workspace_ ||
            !key_workspace_ || !value_workspace_ || !cos_workspace_ ||
            !sin_workspace_ || !context_workspace_ ||
            !normalized_residual_workspace_ || !gate_workspace_ ||
            !up_workspace_ || !down_workspace_) {
            throw std::runtime_error(
                "TransformerBlock::forward_into: GPU workspace is not initialized");
        }
        Tensor& normalized_input = *normalized_input_workspace_;
        Tensor& queries = *query_workspace_;
        Tensor& keys = *key_workspace_;
        Tensor& values = *value_workspace_;
        Tensor& context = *context_workspace_;
        Tensor& normalized_residual = *normalized_residual_workspace_;
        Tensor& gate = *gate_workspace_;
        Tensor& up = *up_workspace_;
        Tensor& down = *down_workspace_;

        rmsnorm_cuda(
            static_cast<const float*>(hidden_states.raw_data()),
            static_cast<float*>(normalized_input.raw_data()),
            static_cast<const float*>(norm1_.gamma().raw_data()),
            static_cast<int>(hidden_dim_), norm1_.epsilon());
        if (parity)
            parity_report("rmsnorm_norm1", host_values(normalized_input),
                          parity_norm1);

        attention_.query_projection().forward_into(normalized_input, queries);
        attention_.key_projection().forward_into(normalized_input, keys);
        attention_.value_projection().forward_into(normalized_input, values);
        std::vector<float> parity_query, parity_key, parity_value;
        if (parity) {
            parity_query = cpu_matmul(parity_norm1, hidden_dim_,
                                      attention_.query_projection().weight());
            parity_key = cpu_matmul(parity_norm1, hidden_dim_,
                                    attention_.key_projection().weight());
            parity_value = cpu_matmul(parity_norm1, hidden_dim_,
                                      attention_.value_projection().weight());
            parity_report("q_proj", host_values(queries), parity_query);
            parity_report("k_proj", host_values(keys), parity_key);
            parity_report("v_proj", host_values(values), parity_value);
        }

        generate_rope_frequencies_cuda(
            static_cast<float*>(cos_workspace_->raw_data()),
            static_cast<float*>(sin_workspace_->raw_data()), head_dim / 2,
            position, head_dim);
        const std::vector<float> parity_cos =
            parity ? host_values(*cos_workspace_) : std::vector<float>();
        const std::vector<float> parity_sin =
            parity ? host_values(*sin_workspace_) : std::vector<float>();
        rope_split_half_cuda(
            static_cast<float*>(queries.raw_data()),
            static_cast<const float*>(cos_workspace_->raw_data()),
            static_cast<const float*>(sin_workspace_->raw_data()), num_heads,
            head_dim);
        rope_split_half_cuda(
            static_cast<float*>(keys.raw_data()),
            static_cast<const float*>(cos_workspace_->raw_data()),
            static_cast<const float*>(sin_workspace_->raw_data()), num_kv_heads,
            head_dim);
        if (parity) {
            cpu_rope_split(parity_query, parity_cos, parity_sin, num_heads,
                           head_dim);
            cpu_rope_split(parity_key, parity_cos, parity_sin, num_kv_heads,
                           head_dim);
            parity_report("rope_q", host_values(queries), parity_query);
            parity_report("rope_k", host_values(keys), parity_key);
        }

        const int page_index = position / block_size;
        const int slot = position % block_size;
        if (page_index < 0 ||
            static_cast<std::size_t>(page_index) >=
                state.physical_blocks.size()) {
            throw std::runtime_error(
                "TransformerBlock::forward_into: no KV page for decode position");
        }
        const int physical_block = state.physical_blocks[page_index];
        const std::size_t token_stride =
            static_cast<std::size_t>(num_kv_heads) *
            static_cast<std::size_t>(head_dim);
        const std::size_t cache_offset =
            (static_cast<std::size_t>(physical_block) *
                 static_cast<std::size_t>(block_size) +
             static_cast<std::size_t>(slot)) * token_stride;
        copy_device(
            keys.raw_data(), block_pool.key_cache() + cache_offset,
            static_cast<std::size_t>(num_kv_heads) *
                static_cast<std::size_t>(head_dim) * sizeof(float),
            "key-cache append");
        copy_device(
            values.raw_data(), block_pool.value_cache() + cache_offset,
            static_cast<std::size_t>(num_kv_heads) *
                static_cast<std::size_t>(head_dim) * sizeof(float),
            "value-cache append");

        const int* device_block_table =
            page_table.get_device_block_table(seq_id);
        paged_attention_decode_cuda(
            static_cast<const float*>(queries.raw_data()),
            block_pool.key_cache(), block_pool.value_cache(),
            device_block_table, static_cast<float*>(context.raw_data()),
            context_len, block_size, num_heads, num_kv_heads, head_dim,
            1.0f / std::sqrt(static_cast<float>(head_dim)));
        std::vector<float> parity_context;
        if (parity) {
            parity_context = cpu_attention(
                parity_query, block_pool, state, num_heads, num_kv_heads,
                head_dim, 1.0f / std::sqrt(static_cast<float>(head_dim)));
            parity_report("paged_attention", host_values(context),
                          parity_context);
        }

        attention_.output_projection().forward_into(context, down);
        std::vector<float> parity_out_proj;
        if (parity) {
            parity_out_proj = cpu_matmul(
                parity_context, hidden_dim_,
                attention_.output_projection().weight());
            parity_report("out_proj", host_values(down), parity_out_proj);
        }
        // Do not overwrite output until every read from hidden_states is done;
        // Generator intentionally supports input/output aliasing here.
        add_inplace_cuda(static_cast<float*>(down.raw_data()),
                         static_cast<const float*>(hidden_states.raw_data()),
                         static_cast<int>(hidden_dim_));

        rmsnorm_cuda(
            static_cast<const float*>(down.raw_data()),
            static_cast<float*>(normalized_residual.raw_data()),
            static_cast<const float*>(norm2_.gamma().raw_data()),
            static_cast<int>(hidden_dim_), norm2_.epsilon());
        std::vector<float> parity_residual;
        std::vector<float> parity_norm2;
        if (parity) {
            parity_residual = parity_input;
            for (std::size_t i = 0; i < parity_residual.size(); ++i)
                parity_residual[i] += parity_out_proj[i];
            parity_norm2 = cpu_rmsnorm(
                parity_residual, host_values(norm2_.gamma()), norm2_.epsilon());
            parity_report("rmsnorm_norm2", host_values(normalized_residual),
                          parity_norm2);
        }

        gate_proj_.forward_into(normalized_residual, gate);
        up_proj_.forward_into(normalized_residual, up);
        std::vector<float> parity_gate, parity_up;
        if (parity) {
            parity_gate = cpu_matmul(
                parity_norm2, hidden_dim_, gate_proj_.weight());
            parity_up = cpu_matmul(
                parity_norm2, hidden_dim_, up_proj_.weight());
            parity_report("gate_proj", host_values(gate), parity_gate);
            parity_report("up_proj", host_values(up), parity_up);
        }
        const int intermediate_dim = static_cast<int>(gate_proj_.output_dim());
        swiglu_inplace_cuda(static_cast<float*>(gate.raw_data()),
                            static_cast<const float*>(up.raw_data()),
                            intermediate_dim);
        std::vector<float> parity_ffn_input;
        if (parity) {
            parity_ffn_input = parity_gate;
            parity_ffn_input.insert(parity_ffn_input.end(),
                                    parity_up.begin(), parity_up.end());
            cpu_swiglu(parity_ffn_input, intermediate_dim);
            parity_report("swiglu", host_values(gate), parity_ffn_input);
        }
        down_proj_.forward_into(gate, context);
        std::vector<float> parity_down;
        if (parity) {
            parity_down = cpu_matmul(
                parity_ffn_input, static_cast<std::size_t>(intermediate_dim),
                down_proj_.weight());
            parity_report("down_proj", host_values(context), parity_down);
            parity_diagnostic_ = false;
        }
        add_inplace_cuda(static_cast<float*>(down.raw_data()),
                         static_cast<const float*>(context.raw_data()),
                         static_cast<int>(hidden_dim_));

        // The caller owns output.  This is the only final copy and does not
        // allocate; it also keeps input/output aliasing safe because all reads
        // from hidden_states completed above.
        copy_device(down.raw_data(), output.raw_data(),
                    hidden_dim_ * sizeof(float), "transformer output");
        return;
    }
#endif
#if !defined(STRATAKV_HAS_CUDA)
    if (output.device() == Device::GPU) {
        throw std::runtime_error(
            "TransformerBlock::forward_into: GPU requires CUDA support");
    }
#endif
    Tensor computed = forward(hidden_states, seq_id, page_table, block_pool,
                              append_tokens);
    for (std::size_t i = 0; i < output.num_elements(); ++i)
        output[i] = computed[i];
}

} // namespace models
} // namespace stratakv
