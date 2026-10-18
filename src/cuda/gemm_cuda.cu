// =============================================================================
//  gemm_cuda.cu  --  Shared-Memory Tiled CUDA GEMM
//
//  Algorithm:
//    Grid  : ceil(M/TILE) x ceil(N/TILE) blocks
//    Block : TILE x TILE threads  (TILE = 32)
//
//  Each block computes one TILE x TILE sub-tile of C.
//  Threads collaboratively stream TILE x TILE sub-tiles of A and B
//  from VRAM into __shared__ memory, eliminating the O(K) redundant
//  global reads a naive kernel would make.
//
//  Shared memory used per block:
//    2 * TILE * TILE * sizeof(float) = 2 * 32 * 32 * 4 = 8 KB
//  Well within the shared-memory budget of the target GPUs.
// =============================================================================

#include "stratakv/cuda_utils.cuh"
#include "stratakv/ops.h"
#include "stratakv/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <cstring>     // memcmp
#include <type_traits>

namespace stratakv {

// ---------------------------------------------------------------------------
//  Kernel constants
// ---------------------------------------------------------------------------
static constexpr int TILE = 32;   // thread-block edge length

// ---------------------------------------------------------------------------
//  gemm_tiled_kernel
//
//  Parameters
//    A  : [M x K] row-major in VRAM
//    B  : [K x N] row-major in VRAM
//    C  : [M x N] row-major in VRAM  (output, pre-zeroed)
//    M, N, K : matrix dimensions (int to avoid sign-extension in index math)
//
//  Each thread computes exactly one element C[row][col].
// ---------------------------------------------------------------------------
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                         float* __restrict__ C,
                                   int M, int N, int K)
{
    // Shared memory tiles. Every global operand loaded by this block is reused
    // by TILE output threads before the next K tile replaces it.
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // This thread's output coordinate in C.
    const int row = static_cast<int>(blockIdx.y) * TILE + static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * TILE + static_cast<int>(threadIdx.x);

    float acc = 0.0f;

    // Iterate over tiles along the K dimension.
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        // ---- Collaborative load: each thread loads one element per tile ----

        // Column index into A this thread is responsible for loading.
        const int a_col = t * TILE + static_cast<int>(threadIdx.x);
        // Row index into B this thread is responsible for loading.
        const int b_row = t * TILE + static_cast<int>(threadIdx.y);

        // Guard: threads at the boundary of non-square / non-multiple-of-TILE
        // matrices load 0.0f so the arithmetic stays correct.
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
                                       ? A[row * K + a_col]
                                       : 0.0f;

        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
                                       ? B[b_row * N + col]
                                       : 0.0f;

        // All threads in the block must have finished loading before any
        // thread starts reading the shared tiles.
        __syncthreads();

        // ---- Compute partial dot product for this tile ----
        // Unroll hint: TILE=32 is known at compile time; nvcc will unroll.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // All threads must finish reading before the next iteration overwrites
        // the shared tiles.
        __syncthreads();
    }

    // Write result -- guard boundary threads.
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

__global__ void add_bias_kernel(float* __restrict__ output,
                                const float* __restrict__ bias,
                                int rows, int columns)
{
    const int index = static_cast<int>(blockIdx.x) * 256 +
        static_cast<int>(threadIdx.x);
    const int total = rows * columns;
    if (index < total)
        output[index] += bias[index % columns];
}

// ---------------------------------------------------------------------------
//  gemm_bf16_weight_kernel
//
//  BF16 weights stay compressed in host and device memory. Each thread owns a
//  pair of adjacent output columns, allowing the cooperative loads of A and B
//  to use __nv_bfloat162. Products are converted to FP32 for accumulation and
//  C remains FP32, matching the inference pipeline's activation precision.
// ---------------------------------------------------------------------------
__device__ float activation_to_float(float value) { return value; }
__device__ float activation_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename Activation>
__global__ void gemm_bf16_weight_kernel(const Activation* __restrict__ A,
                                        const __nv_bfloat16* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K)
{
    __shared__ Activation As[TILE][TILE];
    __shared__ __nv_bfloat16 Bs[TILE][TILE];

    const int row = static_cast<int>(blockIdx.y) * TILE +
        static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * TILE +
        static_cast<int>(threadIdx.x) * 2;
    const int local_col = static_cast<int>(threadIdx.x) * 2;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + local_col;
        if constexpr (std::is_same<Activation, __nv_bfloat16>::value) {
            if (row < M && a_col + 1 < K && (K % 2) == 0) {
            // d_A is cudaMalloc-aligned and a_col is even, so this is a
            // naturally aligned two-element BF16 load. Boundary pairs use
            // scalar loads below to avoid reading past K.
                const __nv_bfloat162 pair =
                    *reinterpret_cast<const __nv_bfloat162*>(
                        A + row * K + a_col);
                As[threadIdx.y][local_col] = pair.x;
                As[threadIdx.y][local_col + 1] = pair.y;
            } else {
                As[threadIdx.y][local_col] =
                    (row < M && a_col < K) ? A[row * K + a_col]
                                           : __float2bfloat16(0.0f);
                As[threadIdx.y][local_col + 1] =
                    (row < M && a_col + 1 < K) ? A[row * K + a_col + 1]
                                               : __float2bfloat16(0.0f);
            }
        } else {
            As[threadIdx.y][local_col] =
                (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
            As[threadIdx.y][local_col + 1] =
                (row < M && a_col + 1 < K) ? A[row * K + a_col + 1]
                                           : 0.0f;
        }

        const int b_row = t * TILE + static_cast<int>(threadIdx.y);
        if (b_row < K && col + 1 < N && (N % 2) == 0) {
            const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(
                B + b_row * N + col);
            Bs[threadIdx.y][local_col] = pair.x;
            Bs[threadIdx.y][local_col + 1] = pair.y;
        } else {
            Bs[threadIdx.y][local_col] =
                (b_row < K && col < N) ? B[b_row * N + col]
                                       : __float2bfloat16(0.0f);
            Bs[threadIdx.y][local_col + 1] =
                (b_row < K && col + 1 < N) ? B[b_row * N + col + 1]
                                           : __float2bfloat16(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const float a = activation_to_float(As[threadIdx.y][k]);
            acc0 += a * __bfloat162float(Bs[k][local_col]);
            acc1 += a * __bfloat162float(Bs[k][local_col + 1]);
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc0;
    if (row < M && col + 1 < N) C[row * N + col + 1] = acc1;
}

// ---------------------------------------------------------------------------
//  matmul_cuda_tiled  --  Host wrapper
//
//  Tensor ownership and transfers are handled outside this wrapper. The
//  persistent path only validates device pointers, launches the kernel, and
//  synchronizes before returning with C resident in VRAM.
// ---------------------------------------------------------------------------
void matmul_cuda_tiled(const Tensor& A, const Tensor& B, Tensor& C)
{
    // ------------------------------------------------------------------
    //  Validation
    // ------------------------------------------------------------------
    if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
        throw std::invalid_argument(
            "matmul_cuda_tiled: all tensors must be 2-D.");
    }

    const int M = static_cast<int>(A.shape()[0]);
    const int K = static_cast<int>(A.shape()[1]);
    const int N = static_cast<int>(B.shape()[1]);

    if (B.shape()[0] != static_cast<size_t>(K)) {
        throw std::invalid_argument(
            "matmul_cuda_tiled: inner dimension mismatch.");
    }
    if (C.shape()[0] != static_cast<size_t>(M) ||
        C.shape()[1] != static_cast<size_t>(N)) {
        throw std::invalid_argument(
            "matmul_cuda_tiled: output tensor C has wrong shape.");
    }

    const bool all_cpu = A.device() == Device::CPU &&
        B.device() == Device::CPU && C.device() == Device::CPU;
    const bool all_gpu = A.device() == Device::GPU &&
        B.device() == Device::GPU && C.device() == Device::GPU;
    if (all_cpu) {
        matmul_cpu_avx2(A, B, C);
        return;
    }
    if (!all_gpu)
        throw std::invalid_argument(
            "matmul_cuda_tiled: A, B, and C must share a device");

    if (C.dtype() != DataType::FP32) {
        throw std::invalid_argument(
            "matmul_cuda_tiled: output tensor must be FP32");
    }
    const bool bf16_inputs = A.dtype() == DataType::BF16 ||
        B.dtype() == DataType::BF16;
    if (bf16_inputs && B.dtype() != DataType::BF16) {
        throw std::invalid_argument(
            "matmul_cuda_tiled: BF16 GEMM requires the right input to be BF16");
    }

    // ------------------------------------------------------------------
    //  Launch directly against persistent Tensor-owned device pointers.
    //
    //  Grid  : each block covers a TILE x TILE patch of C
    //  Block : TILE x TILE threads for FP32, (TILE/2) x TILE for BF16
    //    blockDim.x = col dimension  (N axis)
    //    blockDim.y = row dimension  (M axis)
    // ------------------------------------------------------------------
    const dim3 block(TILE, TILE);
    const dim3 grid(
        (static_cast<unsigned>(N) + TILE - 1) / TILE,
        (static_cast<unsigned>(M) + TILE - 1) / TILE
    );

    if (bf16_inputs) {
        const dim3 bf16_block(TILE / 2, TILE);
        const dim3 bf16_grid(
            (static_cast<unsigned>(N) + TILE - 1) / TILE,
            (static_cast<unsigned>(M) + TILE - 1) / TILE
        );
        if (A.dtype() == DataType::BF16) {
            gemm_bf16_weight_kernel<__nv_bfloat16><<<bf16_grid, bf16_block>>>(
                static_cast<const __nv_bfloat16*>(A.raw_data()),
                static_cast<const __nv_bfloat16*>(B.raw_data()),
                static_cast<float*>(C.raw_data()), M, N, K);
        } else {
            gemm_bf16_weight_kernel<float><<<bf16_grid, bf16_block>>>(
                static_cast<const float*>(A.raw_data()),
                static_cast<const __nv_bfloat16*>(B.raw_data()),
                static_cast<float*>(C.raw_data()), M, N, K);
        }
    } else {
        gemm_tiled_kernel<<<grid, block>>>(
            static_cast<const float*>(A.raw_data()),
            static_cast<const float*>(B.raw_data()),
            static_cast<float*>(C.raw_data()), M, N, K);
    }
    CUDA_CHECK_KERNEL();   // sync + error check
}

void add_bias_cuda(Tensor& output, const Tensor& bias)
{
    if (output.device() != Device::GPU || bias.device() != Device::GPU ||
        output.dtype() != DataType::FP32 || bias.dtype() != DataType::FP32 ||
        output.ndim() != 2 || bias.ndim() != 1 ||
        output.shape()[1] != bias.shape()[0]) {
        throw std::invalid_argument(
            "add_bias_cuda: expected GPU FP32 output [rows, columns] and "
            "GPU FP32 bias [columns]");
    }
    const int rows = static_cast<int>(output.shape()[0]);
    const int columns = static_cast<int>(output.shape()[1]);
    const int total = rows * columns;
    add_bias_kernel<<<(static_cast<unsigned>(total) + 255u) / 256u, 256>>>(
        static_cast<float*>(output.raw_data()),
        static_cast<const float*>(bias.raw_data()), rows, columns);
    CUDA_CHECK_KERNEL();
}

} // namespace stratakv
