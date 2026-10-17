// =============================================================================
//  gemm_cpu.cpp  --  AVX2 cache-tiled GEMM  (Phase 1, StrataKV)
//
//  Algorithm overview:
//    1. Outer cache-tiling loops: partition M, N, K into TILE-sized blocks so
//       the working set for a single micro-kernel fits entirely in L1 cache.
//       TILE = 64 floats == 256 bytes per row, well inside a 32KB L1 D-cache.
//
//    2. Inner AVX2 micro-kernel: for every (i, j) pair inside a tile, we
//       accumulate 8 floats at once using:
//         _mm256_load_ps   -- 256-bit aligned load
//         _mm256_fmadd_ps  -- fused multiply-add (1 instruction, 2 FLOPs)
//         _mm256_store_ps  -- 256-bit aligned store
//
//  Memory layout: all tensors are row-major, 64-byte aligned (Tensor ctor).
// =============================================================================

#include "stratakv/ops.h"
#include "stratakv/tensor.h"

#include <immintrin.h>   // AVX2 + FMA intrinsics
#include <stdexcept>
#include <cstring>       // std::memset / std::memcpy
#include <algorithm>     // std::min
#include <malloc.h>      // _aligned_malloc / _aligned_free

namespace stratakv {

// ---------------------------------------------------------------------------
//  Tuning constants
// ---------------------------------------------------------------------------
//
//  TILE_M / TILE_N / TILE_K control the block sizes fed into the micro-kernel.
//
//    L1 data-cache working set analysis (TILE_M=64, TILE_N=64, TILE_K=64):
//      A tile : 64 * 64 * 4 bytes = 16 KB
//      B tile : 64 * 64 * 4 bytes = 16 KB
//      C tile : 64 * 64 * 4 bytes = 16 KB
//      Total  : 48 KB  --> fits in a typical 64 KB L1 D-cache
//
//  The AVX2 register width is 256 bits = 8 x float32.
//  MICRO_N must be a multiple of 8 so every inner loop is fully vectorised.
// ---------------------------------------------------------------------------

static constexpr size_t TILE_M   = 64;
static constexpr size_t TILE_N   = 64;
static constexpr size_t TILE_K   = 64;
static constexpr size_t MICRO_N  = 8;   // one AVX2 register width

// ---------------------------------------------------------------------------
//  Scratch buffer helper
//  When a tile at the edge of the matrix is smaller than TILE_N, we copy it
//  into a zero-padded, MICRO_N-aligned buffer so the inner AVX2 loop never
//  reads/writes out of bounds.
// ---------------------------------------------------------------------------

struct AlignedScratch {
    float* ptr = nullptr;
    size_t sz  = 0;

    explicit AlignedScratch(size_t elems) : sz(elems) {
        ptr = static_cast<float*>(_aligned_malloc(elems * sizeof(float), 64));
        if (!ptr) throw std::bad_alloc();
        std::memset(ptr, 0, elems * sizeof(float));
    }
    ~AlignedScratch() { if (ptr) _aligned_free(ptr); }

    // Non-copyable, non-movable (local scratch only).
    AlignedScratch(const AlignedScratch&)            = delete;
    AlignedScratch& operator=(const AlignedScratch&) = delete;
};

// ---------------------------------------------------------------------------
//  matmul_cpu_avx2
// ---------------------------------------------------------------------------

void matmul_cpu_avx2(const Tensor& A, const Tensor& B, Tensor& C) {
    // ------------------------------------------------------------------
    //  Validation
    // ------------------------------------------------------------------
    if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
        throw std::invalid_argument(
            "matmul_cpu_avx2: all tensors must be 2-D.");
    }

    const size_t M = A.shape()[0];
    const size_t K = A.shape()[1];
    const size_t N = B.shape()[1];

    if (B.shape()[0] != K) {
        throw std::invalid_argument(
            "matmul_cpu_avx2: inner dimensions mismatch "
            "(A.cols != B.rows).");
    }
    if (C.shape()[0] != M || C.shape()[1] != N) {
        throw std::invalid_argument(
            "matmul_cpu_avx2: output tensor C has wrong shape.");
    }

    // Zero the output matrix.
    std::memset(C.data(), 0, M * N * sizeof(float));

    const float* __restrict a = A.data();
    const float* __restrict b = B.data();
    float*       __restrict c = C.data();

    // Scratch buffers for partial edge tiles (reused across outer loops).
    // Worst case: one TILE_M x TILE_K A-tile and one TILE_K x TILE_N B-tile.
    AlignedScratch a_tile(TILE_M * TILE_K);
    AlignedScratch b_tile(TILE_K * TILE_N);
    AlignedScratch c_tile(TILE_M * TILE_N);

    // ------------------------------------------------------------------
    //  Three-level cache-tiled loops:  M --> N --> K
    //
    //  Iteration order chosen to maximise reuse of the B tile (streamed
    //  once per K-block across all N-blocks) and the C tile (accumulated
    //  across all K-blocks).
    // ------------------------------------------------------------------

    for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
        const size_t tm = std::min(TILE_M, M - i0);   // actual rows in tile

        for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
            const size_t tn = std::min(TILE_N, N - j0);   // actual cols in tile

            // Accumulator tile for C[i0:i0+tm, j0:j0+tn] -- zero it once.
            float* ct = c_tile.ptr;
            std::memset(ct, 0, TILE_M * TILE_N * sizeof(float));

            for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
                const size_t tk = std::min(TILE_K, K - k0);   // actual depth in tile

                // ---- Pack A sub-tile into contiguous scratch (row-major). ----
                // A_tile[ti][tk] = A[i0+ti][k0+tk]
                float* at = a_tile.ptr;
                for (size_t ti = 0; ti < tm; ++ti) {
                    const float* a_row = a + (i0 + ti) * K + k0;
                    float* at_row      = at + ti * TILE_K;
                    std::memcpy(at_row, a_row, tk * sizeof(float));
                    // Pad remainder to zero so AVX2 loads are safe.
                    if (tk < TILE_K) {
                        std::memset(at_row + tk, 0, (TILE_K - tk) * sizeof(float));
                    }
                }

                // ---- Pack B sub-tile into contiguous scratch (row-major). ----
                // B_tile[tk][tj] = B[k0+tk][j0+tj]
                float* bt = b_tile.ptr;
                for (size_t tk2 = 0; tk2 < tk; ++tk2) {
                    const float* b_row = b + (k0 + tk2) * N + j0;
                    float* bt_row      = bt + tk2 * TILE_N;
                    std::memcpy(bt_row, b_row, tn * sizeof(float));
                    if (tn < TILE_N) {
                        std::memset(bt_row + tn, 0, (TILE_N - tn) * sizeof(float));
                    }
                }
                // Pad extra K rows to zero.
                if (tk < TILE_K) {
                    std::memset(bt + tk * TILE_N, 0,
                                (TILE_K - tk) * TILE_N * sizeof(float));
                }

                // ---- AVX2 micro-kernel ------------------------------------------
                //
                //  For each row i in the A tile:
                //    For each column-group j (step=8) in the B tile:
                //      Accumulate C_tile[i][j:j+8] += A_tile[i][k] * B_tile[k][j:j+8]
                //      for all k in [0, TILE_K).
                //
                //  _mm256_fmadd_ps(a_vec, b_vec, c_vec)  computes:
                //      c_vec = a_vec * b_vec + c_vec  (1 instruction on Haswell+)
                // -----------------------------------------------------------------

                for (size_t ti = 0; ti < tm; ++ti) {
                    const float* at_row = at + ti * TILE_K;
                    float*       ct_row = ct + ti * TILE_N;

                    // Process TILE_N columns in steps of 8 (one AVX2 register).
                    for (size_t tj = 0; tj < TILE_N; tj += MICRO_N) {
                        // Load 8 accumulators from the C tile.
                        __m256 c_vec = _mm256_load_ps(ct_row + tj);

                        // Dot-product over the K dimension.
                        for (size_t tk2 = 0; tk2 < TILE_K; ++tk2) {
                            // Broadcast a single scalar from A across all 8 lanes.
                            __m256 a_vec = _mm256_set1_ps(at_row[tk2]);

                            // Load 8 consecutive B values (64-byte aligned base,
                            // offset is a multiple of 32 bytes when tj is a
                            // multiple of 8 floats and bt is 64-byte aligned).
                            __m256 b_vec = _mm256_load_ps(bt + tk2 * TILE_N + tj);

                            // Fused multiply-accumulate.
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        }

                        // Write back 8 accumulators.
                        _mm256_store_ps(ct_row + tj, c_vec);
                    }
                }
            } // k0

            // ---- Scatter C tile back to the global output matrix. ----
            for (size_t ti = 0; ti < tm; ++ti) {
                float*       c_row  = c  + (i0 + ti) * N + j0;
                const float* ct_row = ct + ti * TILE_N;
                std::memcpy(c_row, ct_row, tn * sizeof(float));
            }
        } // j0
    } // i0
}

} // namespace stratakv
