#include "stratakv/tensor.h"

#include "stratakv/ops.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void rejects(Function&& function) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "expected operation to reject");
}
} // namespace

int main() {
    try {
        stratakv::Tensor host({2, 3});
        host[0] = 1.0f;
        host[1] = 2.0f;
        host[2] = 3.0f;
        host[3] = 4.0f;
        host[4] = 5.0f;
        host[5] = 6.0f;
        check(host.device() == stratakv::Device::CPU,
              "default Tensor device is not CPU");
        check(host.to(stratakv::Device::CPU)[5] == 6.0f,
              "CPU-to-CPU Tensor transfer failed");

        stratakv::Tensor matrix({3, 2}, stratakv::DataType::BF16);
        for (std::size_t i = 0; i < 6; ++i)
            matrix[i] = static_cast<float>(i + 1);
        stratakv::Tensor view = matrix.transpose_2d();
        check(view.raw_data() == matrix.raw_data() &&
              view.shape() == std::vector<std::size_t>{2, 3} &&
              view.strides() == std::vector<std::size_t>{1, 2},
              "BF16 transpose must share storage and track matrix strides");
        const stratakv::Tensor copied = view.to(stratakv::Device::CPU);
        check(copied[0] == 1.0f && copied[1] == 3.0f && copied[2] == 5.0f &&
              copied[3] == 2.0f && copied[4] == 4.0f && copied[5] == 6.0f,
              "CPU transfer lost transposed logical indexing");
        stratakv::Tensor cpu_input({1, 2}), cpu_output({1, 3});
        cpu_input[0] = 1.0f;
        cpu_input[1] = 2.0f;
        stratakv::matmul_cpu_avx2(cpu_input, copied, cpu_output);
        check(cpu_output[0] == 5.0f && cpu_output[1] == 11.0f &&
              cpu_output[2] == 17.0f,
              "CPU BF16 GEMM ignored transposed weight strides");
        view[view.get_flat_index({1, 2})] = 42.0f;
        check(matrix[5] == 42.0f && view.transpose_2d().is_contiguous(),
              "BF16 transpose write or double transpose failed");
        rejects([&] { (void)host.transpose_2d(); });

#if defined(STRATAKV_HAS_CUDA)
        stratakv::Tensor right_host({3, 2});
        right_host[0] = 7.0f;
        right_host[1] = 8.0f;
        right_host[2] = 9.0f;
        right_host[3] = 10.0f;
        right_host[4] = 11.0f;
        right_host[5] = 12.0f;

        stratakv::Tensor left = host.to(stratakv::Device::GPU);
        stratakv::Tensor right = right_host.to(stratakv::Device::GPU);
        check(left.device() == stratakv::Device::GPU &&
              right.device() == stratakv::Device::GPU,
              "CPU-to-GPU transfer did not produce GPU tensors");
        check(left.data() == nullptr && left.raw_data() != nullptr,
              "GPU Tensor exposed a host FP32 data pointer");

        stratakv::Tensor product({2, 2}, stratakv::Device::GPU);
        stratakv::matmul_cuda_tiled(left, right, product);
        check(product.device() == stratakv::Device::GPU,
              "direct CUDA GEMM moved its output to the host");
        const stratakv::Tensor product_host =
            product.to(stratakv::Device::CPU);
        check(std::fabs(product_host[0] - 58.0f) < 1.0e-3f &&
              std::fabs(product_host[1] - 64.0f) < 1.0e-3f &&
              std::fabs(product_host[2] - 139.0f) < 1.0e-3f &&
              std::fabs(product_host[3] - 154.0f) < 1.0e-3f,
              "direct persistent-pointer CUDA GEMM result mismatch");

        stratakv::Tensor bf16_left_host({2, 2}, stratakv::DataType::BF16);
        stratakv::Tensor bf16_right_host({2, 2}, stratakv::DataType::BF16);
        bf16_left_host[0] = 1.0f;
        bf16_left_host[1] = 2.0f;
        bf16_left_host[2] = 3.0f;
        bf16_left_host[3] = 4.0f;
        bf16_right_host[0] = 5.0f;
        bf16_right_host[1] = 6.0f;
        bf16_right_host[2] = 7.0f;
        bf16_right_host[3] = 8.0f;
        stratakv::Tensor bf16_product({2, 2}, stratakv::Device::GPU);
        stratakv::matmul_cuda_tiled(
            bf16_left_host.to(stratakv::Device::GPU),
            bf16_right_host.to(stratakv::Device::GPU), bf16_product);
        const stratakv::Tensor bf16_product_host =
            bf16_product.to(stratakv::Device::CPU);
        check(std::fabs(bf16_product_host[0] - 19.0f) < 1.0e-2f &&
              std::fabs(bf16_product_host[1] - 22.0f) < 1.0e-2f &&
              std::fabs(bf16_product_host[2] - 43.0f) < 1.0e-2f &&
              std::fabs(bf16_product_host[3] - 50.0f) < 1.0e-2f,
              "direct persistent-pointer CUDA BF16 GEMM mismatch");

        stratakv::Tensor moved = std::move(product);
        check(moved.device() == stratakv::Device::GPU &&
              moved.raw_data() != nullptr,
              "GPU Tensor move lost device ownership");
        rejects([&] { stratakv::matmul_cuda_tiled(host, right, moved); });
#else
        rejects([&] { (void)host.to(stratakv::Device::GPU); });
#endif

        std::cout << "Phase 8: device Tensor and persistent GEMM checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Phase 8 failure: " << error.what() << '\n';
        return 1;
    }
}
