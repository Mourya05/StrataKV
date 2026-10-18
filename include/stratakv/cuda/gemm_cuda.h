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

// Apply split-half RoPE in place to contiguous FP32 head vectors.
void rope_split_half_cuda(float* tensor, const float* cos_ptr,
                          const float* sin_ptr, int num_heads, int head_dim);
#endif

} // namespace stratakv
