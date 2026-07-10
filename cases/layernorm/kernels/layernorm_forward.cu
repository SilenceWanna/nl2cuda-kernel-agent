#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int THREADS = 256;

__global__ void layernorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int b,
    int d,
    float eps) {
    __shared__ float shared[THREADS];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= b) {
        return;
    }

    const float* x_row = x + static_cast<long long>(row) * d;

    float sum = 0.0f;
    for (int col = tid; col < d; col += THREADS) {
        sum += x_row[col];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float mean = shared[0] / static_cast<float>(d);

    float var_sum = 0.0f;
    for (int col = tid; col < d; col += THREADS) {
        float centered = x_row[col] - mean;
        var_sum += centered * centered;
    }
    shared[tid] = var_sum;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float variance = shared[0] / static_cast<float>(d);
    float rstd = 1.0f / sqrtf(variance + eps);

    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    float* out_row = out + static_cast<long long>(row) * d;
    for (int col = tid; col < d; col += THREADS) {
        float xhat = (x_row[col] - mean) * rstd;
        out_row[col] = xhat * gamma[col] + beta[col];
    }
}

}  // namespace

std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    TORCH_CHECK(x.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "LayerNorm inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                gamma.scalar_type() == torch::kFloat32 &&
                beta.scalar_type() == torch::kFloat32,
                "LayerNorm only supports float32");
    TORCH_CHECK(x.dim() == 2, "LayerNorm X must be 2D");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1,
                "LayerNorm gamma and beta must be 1D");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(),
                "LayerNorm inputs must be contiguous");

    int b = static_cast<int>(x.size(0));
    int d = static_cast<int>(x.size(1));
    TORCH_CHECK(gamma.size(0) == d && beta.size(0) == d,
                "LayerNorm gamma/beta shape mismatch");

    auto out = torch::empty_like(x);
    auto mean = torch::empty({b}, x.options());
    auto rstd = torch::empty({b}, x.options());

    layernorm_forward_kernel<<<b, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        b,
        d,
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out, mean, rstd};
}
