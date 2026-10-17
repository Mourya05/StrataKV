// =============================================================================
//  cuda_utils.cuh  --  CUDA error-checking utilities  (StrataKV)
//
//  Rules:
//    - Wrap EVERY CUDA runtime API call with CUDA_CHECK().
//    - Wrap EVERY kernel launch with CUDA_CHECK_KERNEL() immediately after.
//    - Both macros are zero-overhead in Release builds if NDEBUG is defined
//      AND you set STRATAKV_CUDA_NDEBUG -- otherwise they always check.
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
//  CUDA_CHECK(expr)
//
//  Evaluates `expr` (which must return cudaError_t).
//  On failure: prints file, line, error name + description, then aborts.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            ::fprintf(stderr,                                                  \
                "[CUDA ERROR] %s:%d  code=%d (%s) \"%s\"\n",                  \
                __FILE__, __LINE__,                                            \
                static_cast<int>(_err),                                        \
                ::cudaGetErrorName(_err),                                      \
                ::cudaGetErrorString(_err));                                   \
            ::fflush(stderr);                                                  \
            ::abort();                                                         \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  CUDA_CHECK_KERNEL()
//
//  Must be called IMMEDIATELY after a kernel launch (<<<...>>>).
//  Two steps:
//    1. cudaGetLastError() -- catches launch-configuration errors
//       (e.g. invalid block size) that are reported synchronously.
//    2. cudaDeviceSynchronize() -- waits for the kernel to finish and
//       surfaces any asynchronous execution errors.
//
//  Cost: forces a device sync, so use only in debug/validation paths.
//  In production hot-paths, use CUDA_CHECK_LAUNCH_ONLY() instead.
// ---------------------------------------------------------------------------
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(::cudaGetLastError());                                      \
        CUDA_CHECK(::cudaDeviceSynchronize());                                 \
    } while (0)

// ---------------------------------------------------------------------------
//  CUDA_CHECK_LAUNCH_ONLY()
//
//  Lighter variant: only checks the synchronous launch error.
//  Use when you will call cudaDeviceSynchronize() yourself later.
// ---------------------------------------------------------------------------
#define CUDA_CHECK_LAUNCH_ONLY()                                               \
    do {                                                                       \
        CUDA_CHECK(::cudaGetLastError());                                      \
    } while (0)
