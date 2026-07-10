#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int FAST_D = 1024;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[WARPS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    value = (threadIdx.x < WARPS) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
    }
    return value;
}

__global__ void layernorm_forward_1024_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int b,
    float eps) {
    __shared__ float mean_shared;
    __shared__ float rstd_shared;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= b) {
        return;
    }

    const float4* __restrict__ x4 =
        reinterpret_cast<const float4*>(x + static_cast<long long>(row) * FAST_D);
    float4 vals = x4[tid];
    float sum = vals.x + vals.y + vals.z + vals.w;
    float total = block_reduce_sum(sum);

    if (tid == 0) {
        mean_shared = total * (1.0f / static_cast<float>(FAST_D));
    }
    __syncthreads();

    float mean = mean_shared;
    float dx0 = vals.x - mean;
    float dx1 = vals.y - mean;
    float dx2 = vals.z - mean;
    float dx3 = vals.w - mean;
    float var_sum = dx0 * dx0 + dx1 * dx1 + dx2 * dx2 + dx3 * dx3;
    float var_total = block_reduce_sum(var_sum);

    if (tid == 0) {
        float variance = var_total * (1.0f / static_cast<float>(FAST_D));
        float rstd = rsqrtf(variance + eps);
        rstd_shared = rstd;
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
    __syncthreads();

    const float4* __restrict__ gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* __restrict__ beta4 = reinterpret_cast<const float4*>(beta);
    float4 g = gamma4[tid];
    float4 be = beta4[tid];
    float rstd = rstd_shared;

    float4 result;
    result.x = dx0 * rstd * g.x + be.x;
    result.y = dx1 * rstd * g.y + be.y;
    result.z = dx2 * rstd * g.z + be.z;
    result.w = dx3 * rstd * g.w + be.w;

    float4* __restrict__ out4 =
        reinterpret_cast<float4*>(out + static_cast<long long>(row) * FAST_D);
    out4[tid] = result;
}

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

    auto stream = at::cuda::getCurrentCUDAStream();
    if (d == FAST_D) {
        layernorm_forward_1024_kernel<<<b, THREADS, 0, stream>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            out.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            b,
            static_cast<float>(eps));
    } else {
        layernorm_forward_kernel<<<b, THREADS, 0, stream>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            out.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            b,
            d,
            static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out, mean, rstd};
}
