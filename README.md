# StrataKV

**A zero-dependency, bare-metal C++ and CUDA inference engine for Large Language Models.** 

StrataKV is built from first principles to bypass the massive overhead of standard machine learning frameworks. Designed for maximum hardware utilization, it manually orchestrates memory mapping, IEEE-754 bit-level conversions, and custom GPU kernels to push consumer hardware to its physical memory bandwidth limits. 

## Hardware Performance (GTX 1050 Ti)

On an NVIDIA GTX 1050 Ti (4 GB VRAM, SM 6.1) running a 1.1B parameter model (TinyLlama `BF16` Safetensors):

*   **Decode Throughput:** 31.8 – 33.8 tokens/sec
*   **Model Load Time:** ~4.0 seconds (down from 211+ seconds)
*   **VRAM Utilization:** Saturated via 128-bit vectorized GPU reads.
*   **Numerical Parity:** Validated CPU/GPU parity to $10^{-8}$ absolute difference across all 22 layers.

## Core Architecture

StrataKV achieves its performance profile through strict systems-level optimizations rather than high-level abstractions:

*   **Warp-Collaborative GEMV Decode:** The `M=1` generation loop abandons standard tile-based GEMM approaches. Instead, a custom reduction kernel maps exactly one warp (32 threads) per output feature. It utilizes `__shfl_down_sync` for ultra-fast intra-warp tree reductions.
*   **Vectorized Memory Loads:** VRAM bus saturation is achieved by casting weights to `uint4`, allowing each CUDA thread to load 16 bytes (eight `bfloat16` weights) per clock cycle. 
*   **Zero-Allocation Hot Loop:** Eliminates `cudaMalloc` and `cudaFree` synchronization stalls completely. The `TransformerBlock::forward_into` routine operates strictly on persistent ping-pong VRAM scratchpads allocated once during initialization.
*   **Multi-Threaded Storage Decoding:** The `.safetensors` parser leverages memory-mapped Windows/POSIX file I/O and chunks `F16` and `BF16` conversions across the hardware core count, bypassing single-threaded bottlenecking.
*   **Dynamic Topology Discovery:** Automatically parses vocabulary size, hidden dimensions, layer counts, and Q/KV head topologies directly from the weight shapes, supporting varying Llama-architecture model sizes dynamically.
*   **Portable CPU Fallback:** Seamlessly compiles on Linux, macOS, and Windows with strict POSIX/MSVC memory alignment rules (`posix_memalign` and `_aligned_malloc`) and AVX2 CPU paths for non-CUDA host execution.

## Getting Started

### Prerequisites
*   CMake >= 3.20
*   C++17 compliant compiler (MSVC, GCC, or Clang)
*   CUDA Toolkit (12.x recommended) for GPU execution

### Building from Source

**For CUDA (NVIDIA GPUs):**
```bash
# Configure the build for your specific GPU architecture (e.g., SM 61 for GTX 1050 series)
cmake -S . -B build-cuda -DSTRATAKV_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61

# Build the release executable
cmake --build build-cuda --config Release
```

**For CPU-Only (AVX2):**
```bash
cmake -S . -B build-cpu -DSTRATAKV_CUDA=OFF
cmake --build build-cpu --config Release
```

### CLI Usage

The primary entry point is `stratakv_cli.exe`. It accepts the model path, maximum token limit, generation temperature, seed, and a sequence of prompt token IDs.

```bash
./build-cuda/Release/stratakv_cli.exe <path_to_model.safetensors> <max_tokens> <temperature> <seed> [prompt_tokens...]
```

**Example (Generating 20 tokens with a fixed seed of 15043):**
```bash
./build-cuda/Release/stratakv_cli.exe "models/tinyllama.safetensors" 20 1 15043 29892 2787
```

## Project Structure

*   `src/cuda/`: Custom GPU kernels (`gemm_cuda.cu`, vectorized reductions, paged attention).
*   `src/cpu/`: AVX2-optimized fallback math and aligned tensor memory management.
*   `src/generation/`: The inference loop, KV cache `BlockPool` routing, and logits sampling.
*   `src/io/`: Multi-threaded `.safetensors` memory mapping and schema decoding.
*   `src/models/`: Llama architecture definitions (`transformer_block`, `attention`, `rmsnorm`).
*   `tests/`: End-to-end numeric parity validation against naive CPU reference implementations.

## Author
Developed by Mourya Birru.