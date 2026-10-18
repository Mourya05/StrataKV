#pragma once

#include "stratakv/tensor.h"

namespace stratakv {

#if defined(STRATAKV_HAS_CUDA)
// C = A * B for row-major matrices. BF16 inputs use FP32 accumulation and
// produce an FP32 output tensor; FP32 inputs retain the existing tiled path.
void matmul_cuda_tiled(const Tensor& A, const Tensor& B, Tensor& C);
#endif

} // namespace stratakv
