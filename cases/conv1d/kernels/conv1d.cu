#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kKernelSize = 4;
constexpr int kThreads = 256;
constexpr int kChannelsPerBlock = 8;
constexpr int kWarpsPerBlock = 8;

__global__ void conv1d_forward_kernel(
        const float* __restrict__ x,
        const float* __restrict__ w,
        float* __restrict__ y,
        int64_t total,
        int C,
        int T) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    int t = static_cast<int>(index % T);
    int c = static_cast<int>((index / T) % C);
    const int64_t base = index - t;
    const float* wc = w + static_cast<int64_t>(c) * kKernelSize;

    float value = wc[0] * x[index];
    if (t >= 1) {
        value += wc[1] * x[base + t - 1];
    }
    if (t >= 2) {
        value += wc[2] * x[base + t - 2];
    }
    if (t >= 3) {
        value += wc[3] * x[base + t - 3];
    }
    y[index] = value;
}

__global__ void conv1d_grad_x_kernel(
        const float* __restrict__ grad_y,
        const float* __restrict__ w,
        float* __restrict__ grad_x,
        int64_t total,
        int C,
        int T) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    int t = static_cast<int>(index % T);
    int c = static_cast<int>((index / T) % C);
    const int64_t base = index - t;
    const float* wc = w + static_cast<int64_t>(c) * kKernelSize;

    float value = wc[0] * grad_y[index];
    if (t + 1 < T) {
        value += wc[1] * grad_y[base + t + 1];
    }
    if (t + 2 < T) {
        value += wc[2] * grad_y[base + t + 2];
    }
    if (t + 3 < T) {
        value += wc[3] * grad_y[base + t + 3];
    }
    grad_x[index] = value;
}

__inline__ __device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void conv1d_grad_w_kernel(
        const float* __restrict__ grad_y,
        const float* __restrict__ x,
        float* __restrict__ grad_w,
        int B,
        int C,
        int T) {
    int c = blockIdx.x * kChannelsPerBlock + threadIdx.y;
    if (c >= C) {
        return;
    }

    int lane = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (int b = blockIdx.z; b < B; b += gridDim.z) {
        const int64_t row = (static_cast<int64_t>(b) * C + c) * T;
        for (int t = lane; t < T; t += 32) {
            const float gy = grad_y[row + t];
            sum0 += gy * x[row + t];
            if (t >= 1) {
                sum1 += gy * x[row + t - 1];
            }
            if (t >= 2) {
                sum2 += gy * x[row + t - 2];
            }
            if (t >= 3) {
                sum3 += gy * x[row + t - 3];
            }
        }
    }

    sum0 = warp_reduce_sum(sum0);
    sum1 = warp_reduce_sum(sum1);
    sum2 = warp_reduce_sum(sum2);
    sum3 = warp_reduce_sum(sum3);
    if (lane == 0) {
        float* grad_wc = grad_w + static_cast<int64_t>(c) * kKernelSize;
        atomicAdd(grad_wc + 0, sum0);
        atomicAdd(grad_wc + 1, sum1);
        atomicAdd(grad_wc + 2, sum2);
        atomicAdd(grad_wc + 3, sum3);
    }
}

void validate_inputs(const torch::Tensor& x, const torch::Tensor& w) {
    TORCH_CHECK(x.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(x.dim() == 3, "X must have shape [B, C, T]");
    TORCH_CHECK(w.dim() == 2, "W must have shape [C, 4]");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0, "X dimensions must be positive");
    TORCH_CHECK(w.size(0) == x.size(1), "W channel dimension must match X");
    TORCH_CHECK(w.size(1) == kKernelSize, "W kernel size must be 4");
}

}  // namespace

torch::Tensor conv1d_forward(torch::Tensor x, torch::Tensor w) {
    validate_inputs(x, w);
    x = x.contiguous();
    w = w.contiguous();
    auto y = torch::empty_like(x);

    const int64_t total = x.numel();
    const int C = static_cast<int>(x.size(1));
    const int T = static_cast<int>(x.size(2));
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv1d_forward_kernel<<<blocks, kThreads, 0, stream>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), total, C, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

std::vector<torch::Tensor> conv1d_backward(torch::Tensor grad_y, torch::Tensor x, torch::Tensor w) {
    validate_inputs(x, w);
    TORCH_CHECK(grad_y.is_cuda(), "grad_y must be a CUDA tensor");
    TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32, "grad_y must be float32");
    TORCH_CHECK(grad_y.sizes() == x.sizes(), "grad_y must have the same shape as X");

    grad_y = grad_y.contiguous();
    x = x.contiguous();
    w = w.contiguous();
    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::zeros_like(w);

    const int B = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int T = static_cast<int>(x.size(2));
    const int64_t total = x.numel();
    const int blocks_x = static_cast<int>((total + kThreads - 1) / kThreads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv1d_grad_x_kernel<<<blocks_x, kThreads, 0, stream>>>(
        grad_y.data_ptr<float>(), w.data_ptr<float>(), grad_x.data_ptr<float>(), total, C, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int batch_blocks = B < 8 ? B : 8;
    dim3 block(32, kWarpsPerBlock);
    dim3 grid((C + kChannelsPerBlock - 1) / kChannelsPerBlock, 1, batch_blocks);
    conv1d_grad_w_kernel<<<grid, block, 0, stream>>>(
        grad_y.data_ptr<float>(), x.data_ptr<float>(), grad_w.data_ptr<float>(), B, C, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_w};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("conv1d_forward", &conv1d_forward, "Depthwise causal Conv1d forward (CUDA)");
    module.def("conv1d_backward", &conv1d_backward, "Depthwise causal Conv1d backward (CUDA)");
}
