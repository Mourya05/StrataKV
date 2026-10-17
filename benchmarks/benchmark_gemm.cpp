// =============================================================================
//  benchmarks/benchmark_gemm.cpp  --  StrataKV GEMM Benchmark Runner
//
//  Profiles the AVX2 cache-tiled GEMM across a range of matrix sizes and
//  reports wall-clock time and GFLOP/s for each configuration.
// =============================================================================

#include "stratakv/tensor.h"
#include "stratakv/ops.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace stratakv;

// ---------------------------------------------------------------------------
static void fill_random(Tensor& t, float scale = 1.0f, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    float* p = t.data();
    for (size_t i = 0; i < t.num_elements(); ++i) p[i] = dist(rng);
}

// ---------------------------------------------------------------------------
struct BenchResult {
    size_t M, K, N;
    double ms;
    double gflops;
};

// Run `warmup` iterations to prime caches, then time `iters` iterations.
static BenchResult bench_gemm(size_t M, size_t K, size_t N,
                               int warmup = 2, int iters = 5) {
    Tensor A({M, K}), B({K, N}), C({M, N});
    fill_random(A, 1.0f, 1);
    fill_random(B, 1.0f, 2);

    for (int w = 0; w < warmup; ++w) matmul_cpu_avx2(A, B, C);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) matmul_cpu_avx2(A, B, C);
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double avg_ms   = total_ms / iters;
    const double flops    = 2.0 * M * K * N;
    const double gflops   = flops / (avg_ms * 1e6);

    return {M, K, N, avg_ms, gflops};
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "=========================================\n";
    std::cout << "  StrataKV  --  AVX2 GEMM Benchmark\n";
    std::cout << "=========================================\n\n";

    // Header
    std::cout << std::left
              << std::setw(8)  << "M"
              << std::setw(8)  << "K"
              << std::setw(8)  << "N"
              << std::setw(14) << "Avg time(ms)"
              << std::setw(14) << "GFLOP/s"
              << "\n";
    std::cout << std::string(52, '-') << "\n";

    const std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048};

    for (size_t sz : sizes) {
        const auto r = bench_gemm(sz, sz, sz);
        std::cout << std::fixed << std::setprecision(3)
                  << std::left
                  << std::setw(8)  << r.M
                  << std::setw(8)  << r.K
                  << std::setw(8)  << r.N
                  << std::setw(14) << r.ms
                  << std::setw(14) << r.gflops
                  << "\n";
    }

    std::cout << "\nBenchmark complete.\n";
    return EXIT_SUCCESS;
}
