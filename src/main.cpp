// =============================================================================
//  main.cpp  --  StrataKV Phase 2 + Phase 3
//
//  Phase 2: 1024x1024 SGEMM benchmark -- AVX2 CPU vs 32x32 tiled CUDA GPU.
//  Phase 3: Paged KV Cache -- virtual memory fragmentation smoke test.
//
//  CUDA timing includes full H2D + kernel + D2H round-trip.
// =============================================================================

#include "stratakv/tensor.h"
#include "stratakv/ops.h"
#include "stratakv/memory/block_pool.h"
#include "stratakv/memory/page_table.h"
#include "stratakv/models/transformer_block.h"
#include "stratakv/generation/generator.h"

#include <filesystem>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <exception>


using namespace stratakv;
using namespace stratakv::memory;
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
static void fill_random(Tensor& t, float scale = 1.0f, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    float* p = t.data();
    for (size_t i = 0; i < t.num_elements(); ++i) p[i] = dist(rng);
}

// ---------------------------------------------------------------------------
static float max_abs_diff(const Tensor& ref, const Tensor& cmp) {
    if (ref.num_elements() != cmp.num_elements()) return 1e30f;
    float worst = 0.0f;
    const float* r = ref.data();
    const float* c = cmp.data();
    for (size_t i = 0; i < ref.num_elements(); ++i) {
        float d = std::abs(r[i] - c[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

static void print_sep(char ch = '-', int width = 60) {
    std::cout << std::string(width, ch) << "\n";
}

// ---------------------------------------------------------------------------
//  print_seq_mapping  --  pretty-print one sequence's block list
// ---------------------------------------------------------------------------
static void print_seq_mapping(int seq_id, const PageTable& pt) {
    const SequenceState& s = pt.get_state(seq_id);
    std::cout << "  Seq " << seq_id
              << " (" << std::setw(2) << s.sequence_length << " tokens, "
              << s.physical_blocks.size() << " block"
              << (s.physical_blocks.size() != 1u ? "s" : " ")
              << ")  ->  blocks [ ";
    for (int blk : s.physical_blocks)
        std::cout << blk << " ";
    std::cout << "]\n";
}

static bool run_phase4_smoke_test() {
    using stratakv::Tensor;
    using stratakv::memory::BlockPool;
    using stratakv::memory::PageTable;
    using stratakv::models::TransformerBlock;

    constexpr std::size_t HIDDEN_DIM = 8;
    constexpr std::size_t NUM_HEADS = 2;
    constexpr std::size_t FFN_DIM = 16;
    constexpr int TOTAL_BLOCKS = 8;
    constexpr int BLOCK_SIZE = 4;
    constexpr int SEQ_ID = 1;
    constexpr std::size_t NEW_TOKENS = 5;

    std::cout << "\n";
    print_sep('=');
    std::cout << "  StrataKV -- Phase 4: Transformer Block Smoke Test\n";
    print_sep('=');

    try {
        BlockPool pool(TOTAL_BLOCKS, BLOCK_SIZE);
        PageTable page_table;

        // Sequence 2 claims block 1 between sequence 1's first and second
        // pages. This makes sequence 1's eventual [0, 2] mapping non-contiguous.
        page_table.allocate_sequence(SEQ_ID, pool);
        page_table.allocate_sequence(2, pool);

        TransformerBlock block(
            HIDDEN_DIM, FFN_DIM, NUM_HEADS, TOTAL_BLOCKS, BLOCK_SIZE);
        Tensor hidden_states({NEW_TOKENS, HIDDEN_DIM});
        for (std::size_t i = 0; i < hidden_states.num_elements(); ++i) {
            hidden_states[i] =
                0.01f * static_cast<float>((i % HIDDEN_DIM) + 1u);
        }

        Tensor output = block.forward(
            hidden_states, SEQ_ID, page_table, pool);
        const auto& sequence = page_table.get_state(SEQ_ID);

        const bool shape_ok =
            output.ndim() == 2 &&
            output.shape()[0] == NEW_TOKENS &&
            output.shape()[1] == HIDDEN_DIM;
        const bool length_ok =
            sequence.sequence_length == static_cast<int>(NEW_TOKENS);
        const bool page_count_ok = sequence.physical_blocks.size() == 2u;
        const bool non_contiguous =
            page_count_ok &&
            sequence.physical_blocks[1] != sequence.physical_blocks[0] + 1;
        const bool traversed = block.last_attention_blocks_visited() >= 2u;

        std::cout << "  Forward output shape       : ["
                  << output.shape()[0] << ", " << output.shape()[1]
                  << "]\n";
        std::cout << "  Seq 1 physical pages       : [ ";
        for (int block_id : sequence.physical_blocks) {
            std::cout << block_id << " ";
        }
        std::cout << "]\n";
        std::cout << "  Physical blocks read       : "
                  << block.last_attention_blocks_visited() << "\n";
        std::cout << "  Forward pass               : "
                  << ((shape_ok && length_ok && page_count_ok &&
                       non_contiguous && traversed) ? "PASS" : "FAIL")
                  << "\n";

        page_table.free_sequence(SEQ_ID, pool);
        page_table.free_sequence(2, pool);
        const bool pool_reclaimed =
            pool.free_count() == static_cast<std::size_t>(TOTAL_BLOCKS);
        std::cout << "  KV pool fully reclaimed     : "
                  << (pool_reclaimed ? "PASS" : "FAIL") << "\n";
        print_sep('=');

        return shape_ok && length_ok && page_count_ok &&
               non_contiguous && traversed && pool_reclaimed;
    } catch (const std::exception& error) {
        std::cerr << "  Phase 4 smoke test exception: "
                  << error.what() << "\n";
        print_sep('=');
        return false;
    }
}

void run_tokens(const std::filesystem::path& checkpoint) {
    std::cout << "\n============================================================\n";
    std::cout << "  StrataKV -- Phase 5: Autoregressive Generation\n";
    std::cout << "============================================================\n";
    
    stratakv::generation::GeneratorConfig cfg;
    cfg.hidden_dim = 8;
    cfg.intermediate_dim = 16;
    cfg.num_heads = 2;
    cfg.total_blocks = 32;
    cfg.block_size = 16;
    cfg.vocab_size = 32;
    cfg.eos_token_id = 2;
    
    std::cout << "Loading weights from: " << checkpoint.string() << " ...\n";
    stratakv::generation::Generator engine(cfg, stratakv::io::load_safetensors(checkpoint));
    
    std::cout << "Generating tokens: ";
    const auto tokens = engine.generate({1, 5, 7}, 16);
    for (int id : tokens) std::cout << id << ' ';
    std::cout << "\n============================================================\n";
}

// ===========================================================================
//  main
// ===========================================================================
int main(int argc, char** argv) {
    bool all_ok = true;

    // =========================================================================
    //  PHASE 2: CPU vs GPU GEMM Showdown
    // =========================================================================
    std::cout << "\n";
    print_sep('=');
    std::cout << "  StrataKV -- Phase 2: CPU vs GPU GEMM Showdown\n";
    print_sep('=');

    constexpr size_t M = 1024, K = 1024, N = 1024;
    const double flops = 2.0 * M * K * N;   // MACs -> FLOPs

    std::cout << "\nMatrix dimensions: A[" << M << "x" << K << "] "
              << "* B[" << K << "x" << N << "] = C[" << M << "x" << N << "]\n\n";

    std::cout << "Allocating tensors ... ";
    Tensor A({M, K}), B({K, N}), C_cpu({M, N});
#if defined(STRATAKV_HAS_CUDA)
    Tensor C_gpu({M, N});
#endif
    std::cout << "done.\n";

    std::cout << "Filling A and B with reproducible random data ... ";
    fill_random(A, 1.0f, 42);
    fill_random(B, 1.0f, 43);
    std::cout << "done.\n\n";

    // --- Benchmark 1: AVX2 CPU -----------------------------------------------
    print_sep();
    std::cout << "  [1/2]  CPU  --  AVX2 Cache-Tiled GEMM\n";
    print_sep();

    auto cpu_t0 = Clock::now();
    matmul_cpu_avx2(A, B, C_cpu);
    auto cpu_t1 = Clock::now();

    const double cpu_ms    = Ms(cpu_t1 - cpu_t0).count();
    const double cpu_gflop = flops / (cpu_ms * 1e6);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Wall time  : " << cpu_ms    << " ms\n";
    std::cout << "  Throughput : " << cpu_gflop << " GFLOP/s\n\n";

    // --- Benchmark 2: CUDA GPU -----------------------------------------------
#if defined(STRATAKV_HAS_CUDA)
    print_sep();
    std::cout << "  [2/2]  GPU  --  32x32 Shared-Memory Tiled CUDA GEMM\n";
    std::cout << "         (timing = H2D + kernel + D2H)\n";
    print_sep();

    std::cout << "  Warm-up pass ... ";
    matmul_cuda_tiled(A, B, C_gpu);
    std::cout << "done.\n";

    auto gpu_t0 = Clock::now();
    matmul_cuda_tiled(A, B, C_gpu);
    auto gpu_t1 = Clock::now();

    const double gpu_ms    = Ms(gpu_t1 - gpu_t0).count();
    const double gpu_gflop = flops / (gpu_ms * 1e6);
    const double speedup   = cpu_ms / gpu_ms;

    std::cout << "  Wall time  : " << gpu_ms    << " ms  (incl. PCIe transfer)\n";
    std::cout << "  Throughput : " << gpu_gflop << " GFLOP/s\n\n";

    print_sep();
    std::cout << "  Correctness: CUDA vs CPU (L-inf norm)\n";
    print_sep();

    const float  diff    = max_abs_diff(C_cpu, C_gpu);
    const bool   correct = (diff < 1e-2f);
    all_ok = all_ok && correct;

    std::cout << "  Max |C_cpu - C_gpu| : " << diff << "\n";
    std::cout << "  Result              : " << (correct ? "PASS" : "FAIL") << "\n\n";

    print_sep('=');
    std::cout << "  SUMMARY\n";
    print_sep('=');
    std::cout << std::left
              << "  " << std::setw(30) << "CPU AVX2"
              << std::setw(10) << std::right << cpu_ms    << " ms"
              << "   " << cpu_gflop << " GFLOP/s\n"
              << std::left
              << "  " << std::setw(30) << "GPU CUDA (incl. PCIe)"
              << std::setw(10) << std::right << gpu_ms    << " ms"
              << "   " << gpu_gflop << " GFLOP/s\n";
    print_sep();
    std::cout << "  Speedup (CPU / GPU) : " << std::setprecision(2) << speedup << "x\n";
    print_sep('=');
    std::cout << "\n";
#else
    // CPU-only build
    print_sep('=');
    std::cout << "  SUMMARY (CPU only -- rebuild with -DSTRATAKV_CUDA=ON for GPU)\n";
    print_sep('=');
    std::cout << std::left
              << "  " << std::setw(30) << "CPU AVX2"
              << std::setw(10) << std::right << cpu_ms    << " ms"
              << "   " << cpu_gflop << " GFLOP/s\n";
    print_sep('=');
    std::cout << "\n";
#endif

    // =========================================================================
    //  PHASE 3: Paged KV Cache -- Fragmentation Smoke Test
    //
    //  Simulates concurrent batch inference with three sequences:
    //    Seq 1: 20 tokens  -> 2 blocks  (ceil(20/16) = 2)
    //    Seq 2:  5 tokens  -> 1 block   (ceil(5/16)  = 1)
    //    Seq 3: 35 tokens  -> 3 blocks  (ceil(35/16) = 3)
    //
    //  Sequences are allocated interleaved so that Seq 2's block lands
    //  between Seq 1's two blocks.  This proves the engine stores logically
    //  contiguous token streams in physically non-contiguous memory pages,
    //  exactly as an OS virtual-memory subsystem would.
    //
    //  Expected block layout:
    //    Seq 1 -> [ 0, 2 ]        (non-adjacent: block 1 belongs to Seq 2)
    //    Seq 2 -> [ 1 ]
    //    Seq 3 -> [ 3, 4, 5 ]
    // =========================================================================
    std::cout << "\n";
    print_sep('=');
    std::cout << "  StrataKV -- Phase 3: Paged KV Cache Smoke Test\n";
    print_sep('=');

    constexpr int TOTAL_BLOCKS = 100;
    constexpr int BLOCK_SIZE   = 16;   // tokens per physical block

    BlockPool pool(TOTAL_BLOCKS, BLOCK_SIZE);
    PageTable pt;

    std::cout << "\n  Pool: " << pool.total_blocks()
              << " blocks  x  " << pool.block_size() << " tokens/block\n\n";

    // Step 1 & 2: allocate seq 1 and seq 2 first (claims blocks 0 and 1)
    std::cout << "  [1] allocate_sequence(1)\n";
    pt.allocate_sequence(1, pool);

    std::cout << "  [2] allocate_sequence(2)\n";
    pt.allocate_sequence(2, pool);

    // Step 3: 20 tokens to seq 1 -- block 0 fills at token 16, then block 2
    std::cout << "  [3] append_token x20  ->  Sequence 1\n";
    for (int i = 0; i < 20; ++i)
        pt.append_token(1, pool);

    // Step 4: 5 tokens to seq 2 -- stays in block 1
    std::cout << "  [4] append_token x5   ->  Sequence 2\n";
    for (int i = 0; i < 5; ++i)
        pt.append_token(2, pool);

    // Step 5: new request arrives while the batch is running
    std::cout << "  [5] allocate_sequence(3)\n";
    pt.allocate_sequence(3, pool);

    // Step 6: 35 tokens to seq 3 -- spans blocks 3, 4, 5
    std::cout << "  [6] append_token x35  ->  Sequence 3\n";
    for (int i = 0; i < 35; ++i)
        pt.append_token(3, pool);

    // --- Print physical block layout -----------------------------------------
    std::cout << "\n";
    print_sep();
    std::cout << "  Physical Block Mapping (proof of non-contiguous allocation)\n";
    print_sep();
    print_seq_mapping(1, pt);
    print_seq_mapping(2, pt);
    print_seq_mapping(3, pt);

    // --- Assertions ----------------------------------------------------------
    const SequenceState& s1 = pt.get_state(1);
    const SequenceState& s2 = pt.get_state(2);
    const SequenceState& s3 = pt.get_state(3);

    const bool len_ok =
        s1.sequence_length == 20 &&
        s2.sequence_length == 5  &&
        s3.sequence_length == 35;

    const bool blk_count_ok =
        s1.physical_blocks.size() == 2u &&
        s2.physical_blocks.size() == 1u &&
        s3.physical_blocks.size() == 3u;

    // Core proof: Seq 1's two blocks are NOT adjacent integers because
    // Seq 2 was allocated between them.
    const bool noncontiguous =
        blk_count_ok &&
        (s1.physical_blocks[1] != s1.physical_blocks[0] + 1);

    const int blocks_used = TOTAL_BLOCKS - static_cast<int>(pool.free_count());

    // --- Free all sequences and verify pool recovery -------------------------
    pt.free_sequence(1, pool);
    pt.free_sequence(2, pool);
    pt.free_sequence(3, pool);
    const bool free_ok =
        (pool.free_count() == static_cast<std::size_t>(TOTAL_BLOCKS));

    // --- Report --------------------------------------------------------------
    std::cout << "\n";
    print_sep();
    std::cout << "  Assertion Results\n";
    print_sep();
    std::cout << "  Sequence lengths correct  : " << (len_ok        ? "PASS" : "FAIL") << "\n";
    std::cout << "  Block counts correct      : " << (blk_count_ok  ? "PASS" : "FAIL") << "\n";
    std::cout << "  Non-contiguous layout     : " << (noncontiguous  ? "PASS" : "FAIL") << "\n";
    std::cout << "  Blocks used               : " << blocks_used << " / " << TOTAL_BLOCKS << "\n";
    std::cout << "  Pool fully reclaimed      : " << (free_ok        ? "PASS" : "FAIL") << "\n";
    print_sep('=');
    std::cout << "\n";

    all_ok = all_ok && len_ok && blk_count_ok && noncontiguous && free_ok;

    // --- Phase 4: Transformer Block Smoke Test ---
    all_ok = all_ok && run_phase4_smoke_test();

    // Phase 5: Run the generator if a checkpoint path was passed
    if (argc > 1) {
        try {
            run_tokens(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Phase 5 execution failed: " << e.what() << '\n';
        }
    } else {
        std::cout << "\n[Info] Pass a .safetensors path as an argument to run the Phase 5 generator.\n";
    }
    
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
