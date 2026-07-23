#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kBlockSizeSmall = 128;
constexpr int kBlockSizeLarge = 256;

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template <int BLOCK_SIZE>
__inline__ __device__ void block_reduce_stats(
    float sum,
    float sumsq,
    float inv_D,
    float eps,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    int b,
    float& out_mean,
    float& out_inv_std) {
    __shared__ float shared_sum[32];
    __shared__ float shared_sumsq[32];
    __shared__ float shared_stats[2];

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    constexpr int num_warps = (BLOCK_SIZE + 31) >> 5;

    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);

    if (lane == 0) {
        shared_sum[wid] = sum;
        shared_sumsq[wid] = sumsq;
    }
    __syncthreads();

    if (wid == 0) {
        float total_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        float total_sumsq = (lane < num_warps) ? shared_sumsq[lane] : 0.0f;

        total_sum = warp_reduce_sum(total_sum);
        total_sumsq = warp_reduce_sum(total_sumsq);

        if (lane == 0) {
            const float m = total_sum * inv_D;
            float var = total_sumsq * inv_D - m * m;
            var = var < 0.0f ? 0.0f : var;
            const float inv = rsqrtf(var + eps);

            shared_stats[0] = m;
            shared_stats[1] = inv;
            mean[b] = m;
            inv_std[b] = inv;
        }
    }
    __syncthreads();

    out_mean = shared_stats[0];
    out_inv_std = shared_stats[1];
}

template <int BLOCK_SIZE>
__global__ void layernorm_forward_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    int B,
    int D,
    float eps) {
    const int b = blockIdx.x;
    const int row = b * D;
    const float inv_D = 1.0f / static_cast<float>(D);

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        const float v = x[row + d];
        sum += v;
        sumsq += v * v;
    }

    float m;
    float inv;
    block_reduce_stats<BLOCK_SIZE>(sum, sumsq, inv_D, eps, mean, inv_std, b, m, inv);

    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        const float xhat = (x[row + d] - m) * inv;
        y[row + d] = xhat * gamma[d] + beta[d];
    }
}

template <int BLOCK_SIZE>
__global__ void layernorm_forward_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    int B,
    int D,
    float eps) {
    const int b = blockIdx.x;
    const int row = b * D;
    const int D4 = D >> 2;
    const float inv_D = 1.0f / static_cast<float>(D);

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + row);
    const float4* __restrict__ gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* __restrict__ beta4 = reinterpret_cast<const float4*>(beta);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y + row);

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int i = threadIdx.x; i < D4; i += BLOCK_SIZE) {
        const float4 v = x4[i];

        sum += v.x + v.y + v.z + v.w;
        sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    float m;
    float inv;
    block_reduce_stats<BLOCK_SIZE>(sum, sumsq, inv_D, eps, mean, inv_std, b, m, inv);

    for (int i = threadIdx.x; i < D4; i += BLOCK_SIZE) {
        const float4 xv = x4[i];
        const float4 gv = gamma4[i];
        const float4 bv = beta4[i];

        float4 out;
        out.x = (xv.x - m) * inv * gv.x + bv.x;
        out.y = (xv.y - m) * inv * gv.y + bv.y;
        out.z = (xv.z - m) * inv * gv.z + bv.z;
        out.w = (xv.w - m) * inv * gv.w + bv.w;

        y4[i] = out;
    }
}

template <int BLOCK_SIZE, int D>
__global__ void layernorm_forward_vec4_fixed_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    int B,
    float eps) {
    const int b = blockIdx.x;
    constexpr int D4 = D >> 2;
    const int row = b * D;
    const float inv_D = 1.0f / static_cast<float>(D);

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + row);
    const float4* __restrict__ gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* __restrict__ beta4 = reinterpret_cast<const float4*>(beta);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y + row);

    float sum = 0.0f;
    float sumsq = 0.0f;

#pragma unroll
    for (int i = threadIdx.x; i < D4; i += BLOCK_SIZE) {
        const float4 v = x4[i];

        sum += v.x + v.y + v.z + v.w;
        sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    float m;
    float inv;
    block_reduce_stats<BLOCK_SIZE>(sum, sumsq, inv_D, eps, mean, inv_std, b, m, inv);

#pragma unroll
    for (int i = threadIdx.x; i < D4; i += BLOCK_SIZE) {
        const float4 xv = x4[i];
        const float4 gv = gamma4[i];
        const float4 bv = beta4[i];

        float4 out;
        out.x = (xv.x - m) * inv * gv.x + bv.x;
        out.y = (xv.y - m) * inv * gv.y + bv.y;
        out.z = (xv.z - m) * inv * gv.z + bv.z;
        out.w = (xv.w - m) * inv * gv.w + bv.w;

        y4[i] = out;
    }
}

void check_forward_inputs(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta) {
    TORCH_CHECK(x.is_cuda(), "X must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(beta.is_cuda(), "beta must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "beta must be float32");
    TORCH_CHECK(x.dim() == 2, "X must have shape [B, D]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
    TORCH_CHECK(beta.dim() == 1, "beta must have shape [D]");
    TORCH_CHECK(x.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(gamma.size(0) == x.size(1), "gamma length must equal D");
    TORCH_CHECK(beta.size(0) == x.size(1), "beta length must equal D");
}

inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0x0f) == 0;
}

template <int BLOCK_SIZE>
void launch_layernorm_forward(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    torch::Tensor& y,
    torch::Tensor& mean,
    torch::Tensor& inv_std,
    int B,
    int D,
    float eps,
    bool use_vec4) {
    const dim3 grid(B);
    const dim3 block(BLOCK_SIZE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (use_vec4) {
        layernorm_forward_vec4_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            mean.data_ptr<float>(),
            inv_std.data_ptr<float>(),
            B,
            D,
            eps);
    } else {
        layernorm_forward_scalar_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            mean.data_ptr<float>(),
            inv_std.data_ptr<float>(),
            B,
            D,
            eps);
    }
}

template <int D>
void launch_layernorm_forward_vec4_fixed_256(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    torch::Tensor& y,
    torch::Tensor& mean,
    torch::Tensor& inv_std,
    int B,
    float eps) {
    const dim3 grid(B);
    const dim3 block(kBlockSizeLarge);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layernorm_forward_vec4_fixed_kernel<kBlockSizeLarge, D><<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        B,
        eps);
}

}  // namespace

std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    check_forward_inputs(x, gamma, beta);

    const int B = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));

    auto y = torch::empty_like(x);
    auto row_opts = x.options().dtype(torch::kFloat32);
    auto mean = torch::empty({B}, row_opts);
    auto inv_std = torch::empty({B}, row_opts);

    if (B == 0) {
        return {y, mean, inv_std};
    }

    const bool use_vec4 =
        (D % 4 == 0) &&
        is_aligned_16(x.data_ptr<float>()) &&
        is_aligned_16(gamma.data_ptr<float>()) &&
        is_aligned_16(beta.data_ptr<float>()) &&
        is_aligned_16(y.data_ptr<float>());

    const float eps_f = static_cast<float>(eps);

    if (use_vec4 && D == 768) {
        launch_layernorm_forward_vec4_fixed_256<768>(
            x,
            gamma,
            beta,
            y,
            mean,
            inv_std,
            B,
            eps_f);
    } else if (use_vec4 && D == 1024) {
        launch_layernorm_forward_vec4_fixed_256<1024>(
            x,
            gamma,
            beta,
            y,
            mean,
            inv_std,
            B,
            eps_f);
    } else if (D < 768) {
        launch_layernorm_forward<kBlockSizeSmall>(
            x,
            gamma,
            beta,
            y,
            mean,
            inv_std,
            B,
            D,
            eps_f,
            use_vec4);
    } else {
        launch_layernorm_forward<kBlockSizeLarge>(
            x,
            gamma,
            beta,
            y,
            mean,
            inv_std,
            B,
            D,
            eps_f,
            use_vec4);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {y, mean, inv_std};
}
