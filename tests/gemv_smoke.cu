// Standalone CUDA correctness test; pass --benchmark for the optional timings.
// Link against stratakv in a CUDA-enabled build. CTest can use SKIP_RETURN_CODE
// 77 and TIMEOUT 150. No CUDA events are created by the default correctness run.
#include "stratakv/cuda/gemm_cuda.h"
#include "stratakv/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using stratakv::DataType;
using stratakv::Device;
using stratakv::Tensor;

constexpr int kSkip = 77;
constexpr int kThreads = 256;
constexpr int kWarmup = 5;
constexpr int kTrials = 3;
constexpr int kIterations = 20;
constexpr double kTolerance = 2.0e-4;
constexpr std::size_t kUploadBytes = 1024u * 1024u;
constexpr std::size_t kMaxWeightBytes = 128u * 1024u * 1024u;
constexpr std::uint32_t kActivationSeed = 0x1234abcdU;
constexpr std::uint32_t kChangedActivationSeed = 0x76543210U;
constexpr std::uint32_t kWeightSeed = 0xc001d00dU;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

// Cooperative deadline between bounded units of work. An individual blocking
// CUDA call cannot be interrupted here; use the CTest timeout as the hard limit.
class WalltimeBudget {
public:
    WalltimeBudget()
        : deadline_(std::chrono::steady_clock::now() +
                    std::chrono::seconds(120)) {}

    void check_remaining() const {
        check(std::chrono::steady_clock::now() < deadline_,
              "120-second smoke-test walltime budget exhausted");
    }

private:
    std::chrono::steady_clock::time_point deadline_;
};

struct Shape {
    int m;
    int k;
    int n;
};

std::string describe(Shape shape, DataType activation_type) {
    std::ostringstream text;
    text << (activation_type == DataType::FP32 ? "FP32" : "BF16")
         << " A / BF16 B, M=" << shape.m << " K=" << shape.k
         << " N=" << shape.n;
    return text.str();
}

// Coordinate-dependent, nonsymmetric data, independent of library RNGs. The
// magnitudes are nonzero and FP32 activations need not be BF16-representable.
float sample(std::size_t row, std::size_t column, std::uint32_t seed) {
    std::uint32_t bits = (static_cast<std::uint32_t>(row) + 1u) * 0x9e3779b9u;
    bits ^= (static_cast<std::uint32_t>(column) + 1u) * 0x85ebca6bu;
    bits ^= seed;
    bits ^= bits >> 16u;
    bits *= 0x7feb352du;
    bits ^= bits >> 15u;
    bits *= 0x846ca68bu;
    bits ^= bits >> 16u;
    const float magnitude = static_cast<float>((bits & 0x7ffu) + 33u) / 1024.0f;
    return (bits & 0x80000000u) != 0u ? -magnitude : magnitude;
}

std::uint16_t to_bf16(float value) {
    std::uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "requires FP32 floats");
    std::memcpy(&bits, &value, sizeof(bits));
    // All generated values are finite: round to nearest, ties to even.
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>(bits >> 16u);
}

float from_bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void fill_activation(Tensor& host, Shape shape, std::uint32_t seed) {
    for (int m = 0; m < shape.m; ++m) {
        for (int k = 0; k < shape.k; ++k) {
            host[static_cast<std::size_t>(m) * shape.k + k] = sample(m, k, seed);
        }
    }
}

void fill_weights(Tensor& host, Shape shape) {
    auto* weights = static_cast<std::uint16_t*>(host.raw_data());
    for (int k = 0; k < shape.k; ++k) {
        for (int n = 0; n < shape.n; ++n) {
            weights[static_cast<std::size_t>(k) * shape.n + n] =
                to_bf16(sample(k, n, kWeightSeed));
        }
    }
}

Tensor pack_weights(const Tensor& source, Shape shape) {
    Tensor packed({static_cast<std::size_t>(shape.n),
                   static_cast<std::size_t>(shape.k)}, DataType::BF16);
    auto* destination = static_cast<std::uint16_t*>(packed.raw_data());
    const auto* original = static_cast<const std::uint16_t*>(source.raw_data());
    for (int k = 0; k < shape.k; ++k)
        for (int n = 0; n < shape.n; ++n)
            destination[static_cast<std::size_t>(n) * shape.k + k] =
                original[static_cast<std::size_t>(k) * shape.n + n];
    return packed.transpose_2d();
}

void check_transpose_views() {
    Tensor retained = [] {
        Tensor storage({3, 2}, DataType::BF16);
        for (std::size_t i = 0; i < 6; ++i) storage[i] = static_cast<float>(i + 1);
        Tensor view = storage.transpose_2d();
        check(view.raw_data() == storage.raw_data(), "transpose copied storage");
        return view; // Storage owner must outlive the original Tensor object.
    }();
    const float expected[] = {1, 3, 5, 2, 4, 6};
    check(!retained.is_contiguous(), "transposed matrix lost its strides");
    Tensor device = retained.to(Device::GPU);
    Tensor moved_device = device.to(Device::GPU);
    Tensor host = moved_device.to(Device::CPU);
    Tensor host_copy = host.to(Device::CPU);
    check(device.strides() == retained.strides() &&
          host_copy.strides() == retained.strides(), "transfer lost weight layout");
    for (std::size_t i = 0; i < 6; ++i)
        check(host_copy[i] == expected[i], "transpose/transfer logical value mismatch");
    retained[retained.get_flat_index({1, 2})] = 42.0f;
    check(retained[5] == 42.0f, "transposed indexed write mismatch");
    Tensor original = retained.transpose_2d();
    check(original.is_contiguous() && original[5] == 42.0f,
          "double transpose did not restore original order");
    Tensor invalid({2, 2});
    bool rejected = false;
    try { (void)invalid.transpose_2d(); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "FP32 transpose must reject unsupported view type");
}

void check_initialization_conversion() {
    Tensor host({9, 263});
    for (std::size_t i = 0; i < host.num_elements(); ++i)
        host[i] = sample(i / 263, i % 263, kWeightSeed);
    Tensor source = host.to(Device::GPU);
    Tensor packed(host.shape(), DataType::BF16, Device::GPU);
    stratakv::convert_fp32_to_bf16_cuda(
        static_cast<const float*>(source.raw_data()),
        static_cast<std::uint16_t*>(packed.raw_data()), host.num_elements());
    Tensor converted = packed.transpose_2d().to(Device::CPU);
    const auto* bits = static_cast<const std::uint16_t*>(converted.raw_data());
    for (std::size_t i = 0; i < host.num_elements(); ++i) {
        std::uint32_t expected;
        const float value = host[i];
        std::memcpy(&expected, &value, sizeof(expected));
        check(bits[i] == static_cast<std::uint16_t>(expected >> 16u),
              "initialization converter changed existing BF16 truncation");
        check(converted[(i % 263) * 9 + i / 263] == from_bf16(bits[i]),
              "converted weight transpose has incorrect logical indexing");
    }
}

struct Reference {
    explicit Reference(std::size_t count)
        : values(count, 0.0), absolute_products(count, 0.0) {}
    std::vector<double> values;
    std::vector<double> absolute_products;
};

Reference make_reference(const Tensor& left, const Tensor& right, Shape shape) {
    Reference reference(static_cast<std::size_t>(shape.m) * shape.n);
    const auto* weights = static_cast<const std::uint16_t*>(right.raw_data());
    // Read the stored (already quantized) inputs, not their pre-BF16 values.
    // k/n loop order streams the actual row-major [K,N] weight layout.
    for (int m = 0; m < shape.m; ++m) {
        for (int k = 0; k < shape.k; ++k) {
            const double a = left.value_at(static_cast<std::size_t>(m) * shape.k + k);
            for (int n = 0; n < shape.n; ++n) {
                const double b = from_bf16(weights[static_cast<std::size_t>(k) *
                                                   shape.n + n]);
                const double product = a * b;
                const std::size_t index = static_cast<std::size_t>(m) * shape.n + n;
                reference.values[index] += product;
                reference.absolute_products[index] += std::abs(product);
            }
        }
    }
    return reference;
}

void check_gpu_output(const Tensor& output, const void* original, Shape shape) {
    check(output.device() == Device::GPU && output.dtype() == DataType::FP32,
          "matmul output did not remain GPU FP32");
    check(output.ndim() == 2 &&
          output.shape()[0] == static_cast<std::size_t>(shape.m) &&
          output.shape()[1] == static_cast<std::size_t>(shape.n),
          "matmul changed the output shape");
    check(original != nullptr && output.raw_data() == original &&
          output.data() == nullptr,
          "matmul replaced its persistent output allocation or exposed host data");
    cudaPointerAttributes attributes{};
    cuda_check(cudaPointerGetAttributes(&attributes, output.raw_data()),
               "cudaPointerGetAttributes(output)");
#if CUDART_VERSION >= 10000
    check(attributes.type == cudaMemoryTypeDevice,
          "output pointer is not device memory");
#else
    check(attributes.memoryType == cudaMemoryTypeDevice,
          "output pointer is not device memory");
#endif
    check(attributes.device == 0, "output pointer moved off the selected GPU");
}

std::vector<float> read_output(const Tensor& output) {
    // Explicit synchronization is required even if a debug library build
    // happens to synchronize inside the public wrapper already.
    cuda_check(cudaGetLastError(), "CUDA launch before reference check");
    cuda_check(cudaDeviceSynchronize(), "synchronize before reference check");
    std::vector<float> values(output.num_elements());
    cuda_check(cudaMemcpy(values.data(), output.raw_data(),
                          values.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "read output for reference check");
    return values;
}

void check_close(double actual, double expected, double absolute_products,
                 std::size_t index, const std::string& context) {
    const double tolerance = kTolerance * (1.0 + absolute_products);
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        !std::isfinite(absolute_products) ||
        std::abs(actual - expected) > tolerance) {
        std::ostringstream message;
        message << std::setprecision(17) << context << " at flat output " << index
                << ": actual=" << actual << " expected=" << expected
                << " tolerance=" << tolerance
                << " sum_abs_products=" << absolute_products;
        throw std::runtime_error(message.str());
    }
}

void check_reference(const std::vector<float>& actual, const Reference& reference,
                     const std::string& context) {
    check(actual.size() == reference.values.size(), "reference size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        check_close(actual[i], reference.values[i], reference.absolute_products[i],
                    i, context);
    }
}

void run_correctness_case(Shape shape, DataType activation_type,
                          const WalltimeBudget& budget, bool output_major) {
    const std::string context = describe(shape, activation_type) +
        (output_major ? " output-major weights" : " input-major weights");
    try {
        budget.check_remaining();
        Tensor left_host({static_cast<std::size_t>(shape.m),
                          static_cast<std::size_t>(shape.k)}, activation_type);
        Tensor right_host({static_cast<std::size_t>(shape.k),
                           static_cast<std::size_t>(shape.n)}, DataType::BF16);
        fill_activation(left_host, shape, kActivationSeed);
        fill_weights(right_host, shape);
        Reference reference = make_reference(left_host, right_host, shape);
        Tensor left = left_host.to(Device::GPU);
        Tensor right = output_major ? pack_weights(right_host, shape).to(Device::GPU)
                                    : right_host.to(Device::GPU);
        Tensor output({static_cast<std::size_t>(shape.m),
                       static_cast<std::size_t>(shape.n)}, Device::GPU);
        const void* original = output.raw_data();
        check_gpu_output(output, original, shape);
        // NaN poison catches missing stores, including partial final tiles.
        cuda_check(cudaMemset(output.raw_data(), 0xff,
                              output.num_elements() * sizeof(float)),
                   "poison correctness output");
        for (int call = 0; call < 3; ++call) {
            budget.check_remaining();
            if (call == 2) {
                // Change A in the same allocation to catch stale/cached output.
                fill_activation(left_host, shape, kChangedActivationSeed);
                reference = make_reference(left_host, right_host, shape);
                cuda_check(cudaMemcpy(left.raw_data(), left_host.raw_data(),
                                      left.num_elements() * left.element_size(),
                                      cudaMemcpyHostToDevice),
                           "update persistent activation");
            }
            // The second call deliberately retains C to catch accumulation.
            stratakv::matmul_cuda_tiled(left, right, output);
            check_gpu_output(output, original, shape);
            check_reference(read_output(output), reference,
                            "call " + std::to_string(call + 1));
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(context + ": " + error.what());
    }
}

void run_correctness(const WalltimeBudget& budget) {
    constexpr int ks[] = {1, 7, 8, 31, 255, 256, 257, 263, 264,
                          511, 512, 513, 5632};
    constexpr int ns[] = {1, 7, 8, 9, 15, 16, 17, 256};
    constexpr DataType types[] = {DataType::FP32, DataType::BF16};
    // Exercise the M>1 GEMM dispatch too, including partial and multiple M tiles.
    constexpr Shape controls[] = {{2, 1, 1}, {2, 7, 9}, {3, 31, 16},
                                  {2, 257, 17}, {17, 264, 8},
                                  {33, 513, 256}, {2, 5632, 7}};
    int cases = 0;
    for (bool output_major : {false, true}) {
        for (DataType type : types) {
            for (int k : ks) {
                for (int n : ns) {
                    run_correctness_case({1, k, n}, type, budget, output_major);
                    ++cases;
                }
            }
            for (Shape shape : controls) {
                run_correctness_case(shape, type, budget, output_major);
                ++cases;
            }
        }
    }
    budget.check_remaining();
    std::cout << "PASS: " << cases << " shape/dtype cases, " << cases * 3
              << " public matmul calls; double references, finite outputs, "
                 "persistent GPU output, repeated and changed-input calls\n";
}

// Previous GEMV baseline, kept local to this executable: one thread owns n,
// all K FP32 activations are shared, and B is scalar-loaded as row-major [K,N].
__global__ void baseline_gemv(const float* __restrict__ a,
                             const __nv_bfloat16* __restrict__ b,
                             float* __restrict__ c, int n_count, int k_count) {
    extern __shared__ float input[];
    const int thread = static_cast<int>(threadIdx.x);
    for (int k = thread; k < k_count; k += static_cast<int>(blockDim.x))
        input[k] = a[k];
    __syncthreads();

    const int n = static_cast<int>(blockIdx.x) *
                      static_cast<int>(blockDim.x) + thread;
    if (n >= n_count) return;
    float sum = 0.0f;
    for (int k = 0; k < k_count; ++k)
        sum += input[k] * __bfloat162float(b[k * n_count + n]);
    c[n] = sum;
}

void launch_baseline(const Tensor& left, const Tensor& right, Tensor& output,
                     Shape shape, bool synchronize_each_call) {
    baseline_gemv<<<(shape.n + kThreads - 1) / kThreads, kThreads,
                    static_cast<std::size_t>(shape.k) * sizeof(float)>>>(
        static_cast<const float*>(left.raw_data()),
        static_cast<const __nv_bfloat16*>(right.raw_data()),
        static_cast<float*>(output.raw_data()), shape.n, shape.k);
    cuda_check(cudaGetLastError(), "baseline GEMV launch");
    // Match the public wrapper's optional STRATAKV_SYNC_KERNELS=1 behavior.
    if (synchronize_each_call)
        cuda_check(cudaDeviceSynchronize(), "baseline per-call synchronization");
}

class CudaEvent {
public:
    CudaEvent() { cuda_check(cudaEventCreate(&event_), "cudaEventCreate"); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() {
        if (event_ != nullptr) {
            const cudaError_t status = cudaEventDestroy(event_);
            if (status != cudaSuccess)
                std::cerr << "cudaEventDestroy: " << cudaGetErrorString(status) << '\n';
        }
    }
    cudaEvent_t get() const { return event_; }
    void close() {
        const cudaEvent_t event = event_;
        event_ = nullptr;
        cuda_check(cudaEventDestroy(event), "cudaEventDestroy");
    }

private:
    cudaEvent_t event_ = nullptr;
};

template<class Function>
double milliseconds_per_call(Function&& function, CudaEvent& start,
                             CudaEvent& stop, const WalltimeBudget& budget) {
    budget.check_remaining();
    cuda_check(cudaDeviceSynchronize(), "synchronize before timing");
    cuda_check(cudaEventRecord(start.get()), "record start event");
    for (int iteration = 0; iteration < kIterations; ++iteration)
        function();
    cuda_check(cudaEventRecord(stop.get()), "record stop event");
    cuda_check(cudaEventSynchronize(stop.get()), "synchronize stop event");
    float milliseconds = 0.0f;
    cuda_check(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cudaEventElapsedTime");
    check(std::isfinite(milliseconds) && milliseconds >= 0.0f,
          "invalid CUDA event duration");
    budget.check_remaining();
    return static_cast<double>(milliseconds) / kIterations;
}

// Stream BF16 weights and accumulate the double reference together. The largest
// benchmark holds two 125 MiB device weight layouts for the comparison, at most
// 1 MiB host weight scratch, and O(K+N) activations/references/outputs. Production
// loads just one output-major copy directly from the checkpoint.
Reference prepare_benchmark_weights(const Tensor& left_host, Tensor& right,
                                     Shape shape, const WalltimeBudget& budget) {
    const std::size_t n_count = static_cast<std::size_t>(shape.n);
    const std::size_t k_count = static_cast<std::size_t>(shape.k);
    const std::size_t rows_per_chunk = (std::max)(
        std::size_t{1}, kUploadBytes / (n_count * sizeof(std::uint16_t)));
    std::vector<std::uint16_t> chunk(rows_per_chunk * n_count);
    auto* device_weights = static_cast<std::uint16_t*>(right.raw_data());
    Reference reference(n_count);
    for (std::size_t first = 0; first < k_count; first += rows_per_chunk) {
        budget.check_remaining();
        const std::size_t rows = (std::min)(rows_per_chunk, k_count - first);
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t k = first + row;
            const double a = left_host.data()[k];
            for (std::size_t n = 0; n < n_count; ++n) {
                const std::uint16_t b = to_bf16(sample(k, n, kWeightSeed));
                chunk[row * n_count + n] = b;
                const double product = a * static_cast<double>(from_bf16(b));
                reference.values[n] += product;
                reference.absolute_products[n] += std::abs(product);
            }
        }
        cuda_check(cudaMemcpy(device_weights + first * n_count, chunk.data(),
                              rows * n_count * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "upload benchmark weight chunk");
    }
    return reference;
}

// Benchmark setup only; production loads checkpoint [N,K] rows directly.
__global__ void pack_benchmark_weights(const std::uint16_t* original,
                                        std::uint16_t* packed, int K, int N) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<std::size_t>(K) * N)
        packed[i] = original[(i % K) * static_cast<std::size_t>(N) + i / K];
}

void run_benchmark_case(Shape shape, bool synchronize_each_call,
                        CudaEvent& start, CudaEvent& stop,
                        const WalltimeBudget& budget) {
    const std::string context = describe(shape, DataType::FP32);
    try {
        budget.check_remaining();
        const std::size_t weight_bytes = static_cast<std::size_t>(shape.k) *
                                         shape.n * sizeof(std::uint16_t);
        check(weight_bytes <= kMaxWeightBytes, "benchmark weight exceeds memory cap");
        Tensor left_host({1, static_cast<std::size_t>(shape.k)});
        fill_activation(left_host, shape, kActivationSeed);
        Tensor left = left_host.to(Device::GPU);
        Tensor right({static_cast<std::size_t>(shape.k),
                      static_cast<std::size_t>(shape.n)}, DataType::BF16, Device::GPU);
        const Reference reference = prepare_benchmark_weights(left_host, right,
                                                               shape, budget);
        Tensor packed_storage({static_cast<std::size_t>(shape.n),
                               static_cast<std::size_t>(shape.k)},
                              DataType::BF16, Device::GPU);
        pack_benchmark_weights<<<
            (right.num_elements() + kThreads - 1) / kThreads, kThreads>>>(
                static_cast<const std::uint16_t*>(right.raw_data()),
                static_cast<std::uint16_t*>(packed_storage.raw_data()), shape.k, shape.n);
        cuda_check(cudaGetLastError(), "pack benchmark weights");
        cuda_check(cudaDeviceSynchronize(), "finish benchmark packing before timing");
        Tensor packed = packed_storage.transpose_2d();
        Tensor optimized({1, static_cast<std::size_t>(shape.n)}, Device::GPU);
        Tensor baseline({1, static_cast<std::size_t>(shape.n)}, Device::GPU);
        const void* optimized_pointer = optimized.raw_data();
        const void* baseline_pointer = baseline.raw_data();
        auto public_call = [&] { stratakv::matmul_cuda_tiled(left, packed, optimized); };
        auto baseline_call = [&] {
            launch_baseline(left, right, baseline, shape, synchronize_each_call);
        };
        auto verify = [&] {
            check_gpu_output(optimized, optimized_pointer, shape);
            check_gpu_output(baseline, baseline_pointer, shape);
            const std::vector<float> public_values = read_output(optimized);
            const std::vector<float> baseline_values = read_output(baseline);
            check_reference(public_values, reference, "optimized vs double reference");
            check_reference(baseline_values, reference, "baseline vs double reference");
            for (std::size_t i = 0; i < public_values.size(); ++i) {
                check_close(baseline_values[i], public_values[i],
                            reference.absolute_products[i], i,
                            "baseline vs optimized");
            }
        };
        cuda_check(cudaMemset(optimized.raw_data(), 0xff,
                              optimized.num_elements() * sizeof(float)),
                   "poison optimized benchmark output");
        cuda_check(cudaMemset(baseline.raw_data(), 0xff,
                              baseline.num_elements() * sizeof(float)),
                   "poison baseline benchmark output");
        public_call();
        baseline_call();
        verify(); // Both implementations must agree before any timing is used.
        for (int iteration = 0; iteration < kWarmup; ++iteration) {
            budget.check_remaining();
            public_call();
            baseline_call();
        }
        for (int trial = 0; trial < kTrials; ++trial) {
            double public_ms;
            double baseline_ms;
            // Alternate order to reduce a systematic cache/clock ordering bias.
            if ((trial % 2) == 0) {
                public_ms = milliseconds_per_call(public_call, start, stop, budget);
                baseline_ms = milliseconds_per_call(baseline_call, start, stop, budget);
            } else {
                baseline_ms = milliseconds_per_call(baseline_call, start, stop, budget);
                public_ms = milliseconds_per_call(public_call, start, stop, budget);
            }
            std::cout << "benchmark K=" << shape.k << " N=" << shape.n
                      << " trial=" << trial + 1 << " iterations=" << kIterations
                      << std::fixed << std::setprecision(6)
                      << " optimized_ms/call=" << public_ms
                      << " baseline_ms/call=" << baseline_ms << std::endl;
        }
        verify();
        budget.check_remaining();
    } catch (const std::exception& error) {
        throw std::runtime_error("benchmark " + context + ": " + error.what());
    }
}

void run_benchmarks(const WalltimeBudget& budget) {
    constexpr Shape shapes[] = {{1, 2048, 2048}, {1, 2048, 256},
                                {1, 2048, 5632}, {1, 5632, 2048},
                                {1, 2048, 32000}};
    const char* sync_setting = std::getenv("STRATAKV_SYNC_KERNELS");
    const bool synchronize_each_call = sync_setting != nullptr &&
                                        std::strcmp(sync_setting, "1") == 0;
    std::cout << "Benchmark: FP32 A / BF16 B, M=1; output-major optimized vs "
                 "input-major baseline (packing excluded); " << kWarmup
              << " warmups per implementation, " << kTrials << " trials x "
              << kIterations << " calls; per-call debug sync="
              << (synchronize_each_call ? "on" : "off") << '\n';
    // Constructed only on the --benchmark path, never by correctness checks.
    CudaEvent start;
    CudaEvent stop;
    for (Shape shape : shapes)
        run_benchmark_case(shape, synchronize_each_call, start, stop, budget);
    stop.close();
    start.close();
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool benchmark = false;
        for (int argument = 1; argument < argc; ++argument) {
            if (std::strcmp(argv[argument], "--benchmark") == 0 && !benchmark) {
                benchmark = true;
            } else if (std::strcmp(argv[argument], "--help") == 0) {
                std::cout << "Usage: gemv_smoke [--benchmark]\n";
                return 0;
            } else {
                throw std::runtime_error("Usage: gemv_smoke [--benchmark]");
            }
        }

        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && device_count == 0)) {
            std::cout << "SKIP: no usable CUDA device";
            if (status != cudaSuccess)
                std::cout << " (" << cudaGetErrorString(status) << ')';
            std::cout << '\n';
            return kSkip;
        }
        cuda_check(status, "cudaGetDeviceCount");
        cuda_check(cudaSetDevice(0), "cudaSetDevice(0)");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout << "CUDA device 0: " << properties.name << '\n';
        const WalltimeBudget budget;
        check_transpose_views();
        check_initialization_conversion();
        run_correctness(budget);
        if (benchmark) run_benchmarks(budget);
        cuda_check(cudaDeviceSynchronize(), "final CUDA synchronization");
        budget.check_remaining();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gemv_smoke failure: " << error.what() << '\n';
        return 1;
    }
}
