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
__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float shared[32];

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    constexpr int num_warps = (BLOCK_SIZE + 31) >> 5;

    v = warp_reduce_sum(v);

    if (lane == 0) {
        shared[wid] = v;
    }
    __syncthreads();

    if (wid == 0) {
        float total = (lane < num_warps) ? shared[lane] : 0.0f;
        total = warp_reduce_sum(total);
        if (lane == 0) {
            shared[0] = total;
        }
    }
    __syncthreads();

    return shared[0];
}

template <int BLOCK_SIZE>
__global__ void rmsnorm_forward_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    float* __restrict__ inv_rms,
    int B,
    int D,
    float eps) {
    const int b = blockIdx.x;
    const int row = b * D;
    const float inv_D = 1.0f / static_cast<float>(D);

    float sumsq = 0.0f;
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        const float v = x[row + d];
        sumsq += v * v;
    }

    const float total_sumsq = block_reduce_sum<BLOCK_SIZE>(sumsq);

    __shared__ float shared_inv_rms;
    if (threadIdx.x == 0) {
        shared_inv_rms = rsqrtf(total_sumsq * inv_D + eps);
        inv_rms[b] = shared_inv_rms;
    }
    __syncthreads();

    const float inv = shared_inv_rms;
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        y[row + d] = x[row + d] * inv * gamma[d];
    }
}

template <int BLOCK_SIZE>
__global__ void rmsnorm_forward_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    float* __restrict__ inv_rms,
    int B,
    int D,
    float eps) {
    const int b = blockIdx.x;
    const int row = b * D;
    const int D4 = D >> 2;
    const float inv_D = 1.0f / static_cast<float>(D);

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + row);
    const float4* __restrict__ gamma4 = reinterpret_cast<const float4*>(gamma);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y + row);

    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < D4; i += BLOCK_SIZE) {
        const float4 v = x4[i];
        sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    const float total_sumsq = block_reduce_sum<BLOCK_SIZE>(sumsq);

    __shared__ float shared_inv_rms;
    if (threadIdx.x == 0) {
        shared_inv_rms = rsqrtf(total_sumsq * inv_D + eps);
        inv_rms[b] = shared_inv_rms;
    }
    __syncthreads();

    const float inv = shared_inv_rms;
    for (int i = threadIdx.x; i < D4; i += BLOCK_SIZE) {
        const float4 xv = x4[i];
        const float4 gv = gamma4[i];

        float4 out;
        out.x = xv.x * inv * gv.x;
        out.y = xv.y * inv * gv.y;
        out.z = xv.z * inv * gv.z;
        out.w = xv.w * inv * gv.w;
        y4[i] = out;
    }
}

template <int BLOCK_SIZE, int D>
__global__ void rmsnorm_forward_vec4_fixed_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    float* __restrict__ inv_rms,
    int B,
    float eps) {
    const int b = blockIdx.x;
    constexpr int D4 = D >> 2;
    const int row = b * D;
    const float inv_D = 1.0f / static_cast<float>(D);

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + row);
    const float4* __restrict__ gamma4 = reinterpret_cast<const float4*>(gamma);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y + row);

    float sumsq = 0.0f;

#pragma unroll
    for (int i = threadIdx.x; i < D4; i += BLOCK_SIZE) {
        const float4 v = x4[i];
        sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    const float total_sumsq = block_reduce_sum<BLOCK_SIZE>(sumsq);

    __shared__ float shared_inv_rms;
    if (threadIdx.x == 0) {
        shared_inv_rms = rsqrtf(total_sumsq * inv_D + eps);
        inv_rms[b] = shared_inv_rms;
    }
    __syncthreads();

    const float inv = shared_inv_rms;

#pragma unroll
    for (int i = threadIdx.x; i < D4; i += BLOCK_SIZE) {
        const float4 xv = x4[i];
        const float4 gv = gamma4[i];

        float4 out;
        out.x = xv.x * inv * gv.x;
        out.y = xv.y * inv * gv.y;
        out.z = xv.z * inv * gv.z;
        out.w = xv.w * inv * gv.w;
        y4[i] = out;
    }
}

void check_forward_inputs(
    const torch::Tensor& x,
    const torch::Tensor& gamma) {
    TORCH_CHECK(x.is_cuda(), "X must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(x.dim() == 2, "X must have shape [B, D]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
    TORCH_CHECK(x.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(gamma.size(0) == x.size(1), "gamma length must equal D");
}

inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0x0f) == 0;
}

template <int BLOCK_SIZE>
void launch_rmsnorm_forward(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    torch::Tensor& y,
    torch::Tensor& inv_rms,
    int B,
    int D,
    float eps,
    bool use_vec4) {
    const dim3 grid(B);
    const dim3 block(BLOCK_SIZE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (use_vec4) {
        rmsnorm_forward_vec4_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            y.data_ptr<float>(),
            inv_rms.data_ptr<float>(),
            B,
            D,
            eps);
    } else {
        rmsnorm_forward_scalar_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            y.data_ptr<float>(),
            inv_rms.data_ptr<float>(),
            B,
            D,
            eps);
    }
}

template <int D>
void launch_rmsnorm_forward_vec4_fixed_256(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    torch::Tensor& y,
    torch::Tensor& inv_rms,
    int B,
    float eps) {
    const dim3 grid(B);
    const dim3 block(kBlockSizeLarge);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_forward_vec4_fixed_kernel<kBlockSizeLarge, D><<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        y.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        B,
        eps);
}

}  // namespace

std::vector<torch::Tensor> rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    double eps) {
    check_forward_inputs(x, gamma);

    const int B = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));

    auto y = torch::empty_like(x);
    auto row_opts = x.options().dtype(torch::kFloat32);
    auto inv_rms = torch::empty({B}, row_opts);

    if (B == 0) {
        return {y, inv_rms};
    }

    const bool use_vec4 =
        (D % 4 == 0) &&
        is_aligned_16(x.data_ptr<float>()) &&
        is_aligned_16(gamma.data_ptr<float>()) &&
        is_aligned_16(y.data_ptr<float>());

    const float eps_f = static_cast<float>(eps);

    if (use_vec4 && D == 1024) {
        launch_rmsnorm_forward_vec4_fixed_256<1024>(
            x,
            gamma,
            y,
            inv_rms,
            B,
            eps_f);
    } else if (D < 768) {
        launch_rmsnorm_forward<kBlockSizeSmall>(
            x,
            gamma,
            y,
            inv_rms,
            B,
            D,
            eps_f,
            use_vec4);
    } else {
        launch_rmsnorm_forward<kBlockSizeLarge>(
            x,
            gamma,
            y,
            inv_rms,
            B,
            D,
            eps_f,
            use_vec4);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {y, inv_rms};
}
