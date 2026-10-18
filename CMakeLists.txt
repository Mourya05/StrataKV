cmake_minimum_required(VERSION 3.18)

# Inject the MSVC bypass macro directly into CMake's compiler identification phase
set(CMAKE_CXX_FLAGS_INIT "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
set(CMAKE_CUDA_FLAGS_INIT "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")

# ---------------------------------------------------------
# Phase flag: set STRATAKV_CUDA=ON to include CUDA sources.
# Phase 1 (CPU-only) defaults to OFF so it builds without a
# CUDA-capable GPU or a new enough toolkit.
# ---------------------------------------------------------
option(STRATAKV_CUDA "Enable CUDA sources (Phase 2+)" ON)

# ---------------------------------------------------------
# CUDA 12.2 + MSVC 14.40+ compatibility shim.
# The newer STL headers added a static_assert that fires when
# CUDA < 12.4 is detected.  Suppressing it requires the define
# to reach cl.exe (the host compiler), so it must be forwarded
# through nvcc with -Xcompiler -- a bare -D is not enough.
# Setting CMAKE_CUDA_FLAGS *before* project() ensures this flag
# is present during the compiler-identification step.
# ---------------------------------------------------------
if(STRATAKV_CUDA AND MSVC)
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -Xcompiler /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
        CACHE STRING "" FORCE)
endif()

# Initialise languages -- add CUDA only when requested.
if(STRATAKV_CUDA)
    project(StrataKV LANGUAGES CXX CUDA)
else()
    project(StrataKV LANGUAGES CXX)
endif()

# Target modern standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(STRATAKV_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # sm_61 = Pascal (GTX 10xx) -- default for the installed toolkit.
    # Override with -DCMAKE_CUDA_ARCHITECTURES=75 for Turing,
    # 80 for Ampere, 89 for Ada, etc.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61)
    endif()
endif()

# CUDA-specific compiler options
if(STRATAKV_CUDA AND CMAKE_CUDA_COMPILER_LOADED)
    # Fast math enables hardware-accelerated reciprocal and transcendental ops
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
endif()

# ---------------------------------------------------------
# Core Library: libstratakv
# ---------------------------------------------------------
# Core CPU sources (always compiled)
set(STRATAKV_SOURCES
    src/cpu/tensor.cpp
    src/cpu/gemm_cpu.cpp
    src/memory/block_pool.cpp
    src/memory/page_table.cpp
    src/layers/linear.cpp
    src/layers/rmsnorm.cpp
    src/layers/attention.cpp
    src/models/transformer_block.cpp
    src/io/safetensors.cpp
    src/generation/sampler.cpp
    src/generation/generator.cpp
)

# Optionally append CUDA sources for Phase 2+
if(STRATAKV_CUDA)
    list(APPEND STRATAKV_SOURCES src/cuda/gemm_cuda.cu)
endif()

add_library(stratakv ${STRATAKV_SOURCES})

# Export headers cleanly
target_include_directories(stratakv PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# With CUDA_RESOLVE_DEVICE_SYMBOLS=ON the device-link step (nvcc --dlink)
# runs inside the static library itself, producing a self-contained .lib.
# Pure C++ executables (no .cu sources) can then link it without needing
# their own nvcc device-link pass -- critical for the VS generator on Windows.
if(STRATAKV_CUDA)
    set_target_properties(stratakv PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    # Propagate STRATAKV_HAS_CUDA to all consumers of the library
    # (ops.h and main.cpp gate CUDA declarations behind this symbol)
    target_compile_definitions(stratakv PUBLIC STRATAKV_HAS_CUDA)
endif()

# ---------------------------------------------------------
# Per-language compiler flags
# CRITICAL: use $<COMPILE_LANGUAGE:CXX> so MSVC's /arch:AVX2
# is NEVER forwarded to nvcc (which cannot parse it).
# ---------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(stratakv PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -mavx2 -mfma -Wall -Wextra -Wpedantic>
    )
elseif(MSVC)
    target_compile_options(stratakv PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /arch:AVX2 /W4>
    )
endif()

# ---------------------------------------------------------
# Executables
# ---------------------------------------------------------

# Main CLI entry point
add_executable(stratakv_run src/main.cpp)
target_link_libraries(stratakv_run PRIVATE stratakv)

# Minimal token-in/token-out bridge for scripted inference.
add_executable(stratakv_cli src/cli.cpp)
target_link_libraries(stratakv_cli PRIVATE stratakv)

# Benchmarking runner to profile CPU vs GPU GEMM
add_executable(benchmark_gemm benchmarks/benchmark_gemm.cpp)
target_link_libraries(benchmark_gemm PRIVATE stratakv)

include(CTest)
if(BUILD_TESTING)
    add_executable(phase5_smoke tests/phase5_smoke.cpp)
    target_link_libraries(phase5_smoke PRIVATE stratakv)
    add_test(NAME phase5_smoke
             COMMAND phase5_smoke "${CMAKE_CURRENT_BINARY_DIR}")

    add_executable(phase6_smoke tests/phase6_smoke.cpp)
    target_link_libraries(phase6_smoke PRIVATE stratakv)
    add_test(NAME phase6_smoke
             COMMAND phase6_smoke "${CMAKE_CURRENT_BINARY_DIR}")

    add_executable(phase7_smoke tests/phase7_smoke.cpp)
    target_link_libraries(phase7_smoke PRIVATE stratakv)
    add_test(NAME phase7_smoke
             COMMAND phase7_smoke "${CMAKE_CURRENT_BINARY_DIR}")
endif()
