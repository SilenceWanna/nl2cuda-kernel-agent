#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kItemsPerThread = 16;
constexpr int kColumns = kThreads * kItemsPerThread;

using BlockScan = cub::BlockScan<float, kThreads>;

union SharedStorage {
    typename BlockScan::TempStorage scan;
    float transpose[kColumns];
};

template <bool kReverse>
__global__ void __launch_bounds__(kThreads, 4) scan_4096_kernel(
    const float* __restrict__ input,
    float* __restrict__ output) {
    __shared__ SharedStorage storage;

    const int tid = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * kColumns;
    float values[kItemsPerThread];

    // Striped global accesses are coalesced. The shared-memory transpose then
    // gives BlockScan the blocked item order it expects from each thread.
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const int logical = item * kThreads + tid;
        const int column = kReverse ? (kColumns - 1 - logical) : logical;
        storage.transpose[logical] = input[row_offset + column];
    }
    __syncthreads();

#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        values[item] = storage.transpose[tid * kItemsPerThread + item];
    }
    __syncthreads();

    BlockScan(storage.scan).InclusiveSum(values, values);
    __syncthreads();

#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        storage.transpose[tid * kItemsPerThread + item] = values[item];
    }
    __syncthreads();

#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const int logical = item * kThreads + tid;
        const int column = kReverse ? (kColumns - 1 - logical) : logical;
        output[row_offset + column] = storage.transpose[logical];
    }
}

void check_input(const torch::Tensor& input, const char* operation) {
    TORCH_CHECK(input.is_cuda(), operation, " expects a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        operation,
        " only supports float32");
    TORCH_CHECK(input.is_contiguous(), operation, " expects a contiguous tensor");
    TORCH_CHECK(input.dim() == 2, operation, " expects a 2D tensor");
    TORCH_CHECK(
        input.size(1) == kColumns,
        operation,
        " expects last dimension to be 4096");
    TORCH_CHECK(input.size(0) > 0, operation, " expects at least one row");
}

}  // namespace

torch::Tensor scan_forward(torch::Tensor input) {
    check_input(input, "scan_forward");
    auto output = torch::empty_like(input);
    const int rows = static_cast<int>(input.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    scan_4096_kernel<false><<<rows, kThreads, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor scan_backward(torch::Tensor grad_output) {
    check_input(grad_output, "scan_backward");
    auto grad_input = torch::empty_like(grad_output);
    const int rows = static_cast<int>(grad_output.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    scan_4096_kernel<true><<<rows, kThreads, 0, stream>>>(
        grad_output.data_ptr<float>(), grad_input.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("scan_forward", &scan_forward, "Inclusive scan forward (CUDA)");
    module.def("scan_backward", &scan_backward, "Inclusive scan backward (CUDA)");
}
