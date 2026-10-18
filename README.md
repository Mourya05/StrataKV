# StrataKV

A zero-dependency, bare-metal C++17 and CUDA LLM inference engine built entirely from first principles. StrataKV implements the Llama-2 architecture from scratch, featuring a custom Paged KV Cache, native `bfloat16` memory aliasing, and bespoke cooperative-load CUDA matrix multiplication kernels. 

It relies on strictly zero external tensor libraries—no cuBLAS, no LibTorch, and no pre-built computation graphs. 

## Core Architecture

* **Custom Tensor Engine:** Hand-rolled C++ memory allocators and multi-dimensional tensor representations with explicit `Device::CPU` and `Device::GPU` ownership states.
* **BF16 Memory Aliasing:** Eliminates upcast bottlenecks by mapping 2.2 GB Safetensors directly from disk via `mmap`, retaining `bfloat16` payloads and dispatching directly to native `__nv_bfloat16` and `__nv_bfloat162` vectorized CUDA kernels.
* **Paged KV Cache:** Manages autoregressive memory fragmentation using a custom `PageTable` and `BlockPool` architecture, mirroring the foundational concepts of vLLM.
* **Split-Half RoPE:** Implements Rotary Positional Embeddings natively in C++ for accurate spatial attention scaling.
* **Stochastic Sampling Suite:** Replaces greedy argmax with a custom C++ probability sampler implementing Temperature scaling, Top-K, and Top-P (nucleus) filtering via stable softmax and `std::mt19937` categorical distributions.

## Hardware & Performance Diagnostics

StrataKV was designed to hit fundamental hardware limits and expose the realities of GPU inference architectures. 

* **The Compute Engine:** The custom CUDA GEMM implementation uses 32x32 shared-memory cache tiling and collaborative memory loads, achieving **~169 GFLOP/s** single-kernel throughput on consumer SM 6.1 hardware.
* **The PCIe Bottleneck:** As a hybrid CPU/GPU engine, current generation throughput sits at **~0.87 Tokens/Second**. Profiling reveals this is a strict architectural PCIe bandwidth limit. While the matrix math executes in ~7.2ms in VRAM, the engine's host-side transformer loops (RMSNorm, RoPE, SwiGLU, Paged Attention) require moving data across the host-to-device boundary per layer.
* **Future Work:** Full zero-copy generation requires porting the remaining host loops into device-readable CUDA kernels to achieve persistent VRAM state.

## Build System

Built natively for Windows MSVC and NVCC. The `CMakeLists.txt` pipeline strictly enforces hardware optimization flags:
* **Host (MSVC):** `/O2`, `/fp:fast`, `/arch:AVX2`
* **Device (NVCC):** `-O3`, `-use_fast_math`, `-Xptxas=-O3`

### Build Instructions
```bash
mkdir build && cd build
cmake .. -DSTRATAKV_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## Execution
The engine uses a Python token-bridge to handle the Hugging Face tokenizer before passing IDs to the C++ core:

*Note: Download the `model.safetensors` file from the `TinyLlama-1.1B-Chat-v1.0` Hugging Face repository and place it in the project root before executing.*

```bash
python runner.py --temp 0.8 --top-k 40 --top-p 0.95
```

## Author
Mourya Birru