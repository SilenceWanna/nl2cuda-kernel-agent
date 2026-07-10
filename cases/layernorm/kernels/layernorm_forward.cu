#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        shared[wid] = v;
    }
    __syncthreads();

    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
    }
    __syncthreads();
    return v;
}

__global__ void layernorm_forward_kernel(
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
    if (b >= B) {
        return;
    }

    const int row = b * D;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sum += x[row + d];
    }
    sum = block_reduce_sum(sum);
    const float m = sum / static_cast<float>(D);

    float ss = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float c = x[row + d] - m;
        ss += c * c;
    }
    ss = block_reduce_sum(ss);
    const float inv = rsqrtf(ss / static_cast<float>(D) + eps);

    if (threadIdx.x == 0) {
        mean[b] = m;
        inv_std[b] = inv;
    }

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float xhat = (x[row + d] - m) * inv;
        y[row + d] = xhat * gamma[d] + beta[d];
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

    const dim3 grid(B);
    const dim3 block(kBlockSize);
    layernorm_forward_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        B,
        D,
        static_cast<float>(eps));

    return {y, mean, inv_std};
}
