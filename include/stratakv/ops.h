#pragma once

#include "stratakv/tensor.h"
#if defined(STRATAKV_HAS_CUDA)
#include "stratakv/cuda/gemm_cuda.h"
#endif

namespace stratakv {

// ============================================================
//  CPU GEMM  --  AVX2 cache-tiled matrix multiplication
//
//  C = A * B   (A: M x K,  B: K x N,  C: M x N)
//  All tensors must be 2-D, inner dims must match, C pre-allocated.
//  All data pointers are guaranteed 64-byte aligned by Tensor ctor.
// ============================================================
void matmul_cpu_avx2(const Tensor& A, const Tensor& B, Tensor& C);

// ============================================================
//  CUDA GEMM  --  32x32 shared-memory tiled kernel
//
//  Same mathematical contract as matmul_cpu_avx2.
//  Timing includes host<->device memory transfers so the number
//  is a fair real-world latency (not just kernel time).
//
//  Only declared when STRATAKV_HAS_CUDA is defined by CMake.
// ============================================================
} // namespace stratakv
