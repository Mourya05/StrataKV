#pragma once

#include "stratakv/tensor.h"

namespace stratakv {

#if defined(STRATAKV_HAS_CUDA)
// C = A * B for row-major matrices. GPU tensors are computed in place without
// staging allocations or host/device transfers. All-CPU inputs are delegated
// to the CPU GEMM for compatibility with existing callers.
void matmul_cuda_tiled(const Tensor& A, const Tensor& B, Tensor& C);

// Add a one-dimensional FP32 bias to every row of a GPU output tensor.
void add_bias_cuda(Tensor& output, const Tensor& bias);

// Apply RMSNorm to one FP32 device vector.
void rmsnorm_cuda(const float* input, float* output, const float* weight,
                  int hidden_size, float epsilon);

// Apply SwiGLU in place to a merged [gate, up] FP32 device vector.
void swiglu_cuda(float* tensor, int hidden_size);

// Apply interleaved RoPE in place to contiguous FP32 head vectors.
void rope_interleaved_cuda(float* tensor, const float* cos_ptr,
                           const float* sin_ptr, int num_heads, int head_dim);
void generate_rope_frequencies_cuda(float* cos_ptr, float* sin_ptr,
                                    int half_dim, int position, int head_dim);
void add_inplace_cuda(float* target, const float* source, int count);
void embedding_lookup_cuda(const float* table, int token_id, int hidden_size,
                           float* output);

// Decode one token against paged FP32 KV caches with grouped-query attention.
void paged_attention_decode_cuda(
    const float* query, const float* key_cache, const float* value_cache,
    const int* block_table, float* output, int context_len, int block_size,
    int num_q_heads, int num_kv_heads, int head_dim, float scale);
#endif

} // namespace stratakv
