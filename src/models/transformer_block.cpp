#include "stratakv/models/transformer_block.h"
#include "stratakv/ops.h"

#include <cmath>
#include <stdexcept>
#include <string>

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
        if (hidden_states.shape()[0] != 1) {
            throw std::invalid_argument(
                "TransformerBlock::forward: GPU decode requires one token");
        }
        if (hidden_states.dtype() != DataType::FP32 ||
            norm1_.gamma().dtype() != DataType::FP32 ||
            norm2_.gamma().dtype() != DataType::FP32 ||
            norm1_.gamma().device() != Device::GPU ||
            norm2_.gamma().device() != Device::GPU ||
            attention_.query_projection().weight().device() != Device::GPU ||
            attention_.key_projection().weight().device() != Device::GPU ||
            attention_.value_projection().weight().device() != Device::GPU ||
            attention_.output_projection().weight().device() != Device::GPU ||
            gate_proj_.weight().device() != Device::GPU ||
            up_proj_.weight().device() != Device::GPU ||
            down_proj_.weight().device() != Device::GPU) {
            throw std::invalid_argument(
                "TransformerBlock::forward: GPU input requires all weights "
                "and FP32 norm parameters to be GPU-resident");
        }
        if (block_pool.device() != Device::GPU ||
            block_pool.key_cache() == nullptr ||
            block_pool.value_cache() == nullptr ||
            block_pool.num_kv_heads() !=
                static_cast<int>(attention_.num_kv_heads()) ||
            block_pool.head_dim() !=
                static_cast<int>(attention_.head_dim())) {
            throw std::invalid_argument(
                "TransformerBlock::forward: GPU BlockPool configuration does "
                "not match attention");
        }
        if (!page_table.has_sequence(seq_id)) {
            throw std::runtime_error(
                "TransformerBlock::forward: seq_id is not registered");
        }

        const memory::SequenceState& initial_state =
            page_table.get_state(seq_id);
        if (initial_state.sequence_length <= 0 && !append_tokens) {
            throw std::invalid_argument(
                "TransformerBlock::forward: cannot decode without context");
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

        Tensor normalized_input({1, hidden_dim_}, DataType::FP32,
                                Device::GPU);
        rmsnorm_cuda(
            static_cast<const float*>(hidden_states.raw_data()),
            static_cast<float*>(normalized_input.raw_data()),
            static_cast<const float*>(norm1_.gamma().raw_data()),
            static_cast<int>(hidden_dim_), norm1_.epsilon());

        Tensor queries = attention_.query_projection().forward(normalized_input);
        Tensor keys = attention_.key_projection().forward(normalized_input);
        Tensor values = attention_.value_projection().forward(normalized_input);

        Tensor cos_values({static_cast<std::size_t>(head_dim / 2)},
                          DataType::FP32, Device::GPU);
        Tensor sin_values({static_cast<std::size_t>(head_dim / 2)},
                          DataType::FP32, Device::GPU);
        generate_rope_frequencies_cuda(
            static_cast<float*>(cos_values.raw_data()),
            static_cast<float*>(sin_values.raw_data()), head_dim / 2,
            position, head_dim);
        rope_split_half_cuda(
            static_cast<float*>(queries.raw_data()),
            static_cast<const float*>(cos_values.raw_data()),
            static_cast<const float*>(sin_values.raw_data()), num_heads,
            head_dim);
        rope_split_half_cuda(
            static_cast<float*>(keys.raw_data()),
            static_cast<const float*>(cos_values.raw_data()),
            static_cast<const float*>(sin_values.raw_data()), num_kv_heads,
            head_dim);

        const int page_index = position / block_size;
        const int slot = position % block_size;
        if (page_index < 0 ||
            static_cast<std::size_t>(page_index) >=
                state.physical_blocks.size()) {
            throw std::runtime_error(
                "TransformerBlock::forward: no KV page for decode position");
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
            keys.raw_data(),
            block_pool.key_cache() + cache_offset,
            static_cast<std::size_t>(num_kv_heads) *
                static_cast<std::size_t>(head_dim) * sizeof(float),
            "key-cache append");
        copy_device(
            values.raw_data(),
            block_pool.value_cache() + cache_offset,
            static_cast<std::size_t>(num_kv_heads) *
                static_cast<std::size_t>(head_dim) * sizeof(float),
            "value-cache append");

        const int* device_block_table =
            page_table.get_device_block_table(seq_id);
        Tensor context({1, hidden_dim_}, DataType::FP32, Device::GPU);
        paged_attention_decode_cuda(
            static_cast<const float*>(queries.raw_data()),
            block_pool.key_cache(), block_pool.value_cache(),
            device_block_table, static_cast<float*>(context.raw_data()),
            context_len, block_size, num_heads, num_kv_heads, head_dim,
            1.0f / std::sqrt(static_cast<float>(head_dim)));

        Tensor attention_output =
            attention_.output_projection().forward(context);
        Tensor residual({1, hidden_dim_}, DataType::FP32, Device::GPU);
        copy_device(hidden_states.raw_data(), residual.raw_data(),
                    hidden_dim_ * sizeof(float), "attention residual copy");
        add_inplace_cuda(static_cast<float*>(residual.raw_data()),
                         static_cast<const float*>(attention_output.raw_data()),
                         static_cast<int>(hidden_dim_));

        Tensor normalized_residual({1, hidden_dim_}, DataType::FP32,
                                   Device::GPU);
        rmsnorm_cuda(
            static_cast<const float*>(residual.raw_data()),
            static_cast<float*>(normalized_residual.raw_data()),
            static_cast<const float*>(norm2_.gamma().raw_data()),
            static_cast<int>(hidden_dim_), norm2_.epsilon());

        Tensor gate = gate_proj_.forward(normalized_residual);
        Tensor up = up_proj_.forward(normalized_residual);
        const int intermediate_dim = static_cast<int>(gate_proj_.output_dim());
        Tensor merged({1, static_cast<std::size_t>(intermediate_dim) * 2u},
                      DataType::FP32, Device::GPU);
        copy_device(gate.raw_data(), merged.raw_data(),
                    static_cast<std::size_t>(intermediate_dim) * sizeof(float),
                    "gate projection pack");
        copy_device(
            up.raw_data(),
            static_cast<float*>(merged.raw_data()) + intermediate_dim,
            static_cast<std::size_t>(intermediate_dim) * sizeof(float),
            "up projection pack");
        swiglu_cuda(static_cast<float*>(merged.raw_data()), intermediate_dim);

        Tensor ffn_input({1, static_cast<std::size_t>(intermediate_dim)},
                         DataType::FP32, Device::GPU);
        copy_device(merged.raw_data(), ffn_input.raw_data(),
                    static_cast<std::size_t>(intermediate_dim) * sizeof(float),
                    "SwiGLU output unpack");
        Tensor ffn_output = down_proj_.forward(ffn_input);
        add_inplace_cuda(static_cast<float*>(residual.raw_data()),
                         static_cast<const float*>(ffn_output.raw_data()),
                         static_cast<int>(hidden_dim_));
        return residual;
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

} // namespace models
} // namespace stratakv
