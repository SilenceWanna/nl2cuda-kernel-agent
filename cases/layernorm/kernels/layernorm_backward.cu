#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps);

namespace {

constexpr int kBlockSize = 256;
constexpr int kColTile = 32;
constexpr int kRowTile = 8;

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

__global__ void layernorm_backward_x_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ grad_x,
    int B,
    int D) {
    const int b = blockIdx.x;
    if (b >= B) {
        return;
    }

    const int row = b * D;
    const float m = mean[b];
    const float inv = inv_std[b];

    // dX = inv_std * (v - mean(v) - xhat * mean(v*xhat)),
    // where v = dY * gamma and xhat = (X - mean) * inv_std.
    float sum_v = 0.0f;
    float sum_v_xhat = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float xhat = (x[row + d] - m) * inv;
        const float v = grad_y[row + d] * gamma[d];
        sum_v += v;
        sum_v_xhat += v * xhat;
    }

    sum_v = block_reduce_sum(sum_v);
    sum_v_xhat = block_reduce_sum(sum_v_xhat);

    const float mean_v = sum_v / static_cast<float>(D);
    const float mean_v_xhat = sum_v_xhat / static_cast<float>(D);

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float xhat = (x[row + d] - m) * inv;
        const float v = grad_y[row + d] * gamma[d];
        grad_x[row + d] = inv * (v - mean_v - xhat * mean_v_xhat);
    }
}

__global__ void zero_vector_kernel(float* __restrict__ a, float* __restrict__ b, int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        a[i] = 0.0f;
        b[i] = 0.0f;
    }
}

// 2D row/column tiled partial reduction for dgamma/dbeta.
// Thread x maps to contiguous columns, so reads of X/dY are coalesced.
// Different row tiles accumulate with atomics into the final [D] vectors.
__global__ void layernorm_backward_param_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int B,
    int D) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_start = blockIdx.y * kRowTile;

    if (col >= D) {
        return;
    }

    float sum_gamma = 0.0f;
    float sum_beta = 0.0f;

#pragma unroll
    for (int r = 0; r < kRowTile; ++r) {
        const int b = row_start + r;
        if (b < B) {
            const int idx = b * D + col;
            const float gy = grad_y[idx];
            const float xhat = (x[idx] - mean[b]) * inv_std[b];
            sum_gamma += gy * xhat;
            sum_beta += gy;
        }
    }

    atomicAdd(grad_gamma + col, sum_gamma);
    atomicAdd(grad_beta + col, sum_beta);
}

void check_backward_inputs(
    const torch::Tensor& grad_y,
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& mean,
    const torch::Tensor& inv_std) {
    TORCH_CHECK(grad_y.is_cuda(), "grad_y must be CUDA");
    TORCH_CHECK(x.is_cuda(), "X must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(mean.is_cuda(), "mean must be CUDA");
    TORCH_CHECK(inv_std.is_cuda(), "inv_std must be CUDA");
    TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32, "grad_y must be float32");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(mean.scalar_type() == torch::kFloat32, "mean must be float32");
    TORCH_CHECK(inv_std.scalar_type() == torch::kFloat32, "inv_std must be float32");
    TORCH_CHECK(grad_y.dim() == 2, "grad_y must have shape [B, D]");
    TORCH_CHECK(x.dim() == 2, "X must have shape [B, D]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
    TORCH_CHECK(mean.dim() == 1, "mean must have shape [B]");
    TORCH_CHECK(inv_std.dim() == 1, "inv_std must have shape [B]");
    TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(mean.is_contiguous(), "mean must be contiguous");
    TORCH_CHECK(inv_std.is_contiguous(), "inv_std must be contiguous");
    TORCH_CHECK(grad_y.size(0) == x.size(0), "grad_y B must match X B");
    TORCH_CHECK(grad_y.size(1) == x.size(1), "grad_y D must match X D");
    TORCH_CHECK(gamma.size(0) == x.size(1), "gamma length must equal D");
    TORCH_CHECK(mean.size(0) == x.size(0), "mean length must equal B");
    TORCH_CHECK(inv_std.size(0) == x.size(0), "inv_std length must equal B");
}

}  // namespace

std::vector<torch::Tensor> layernorm_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor mean,
    torch::Tensor inv_std) {
    check_backward_inputs(grad_y, x, gamma, mean, inv_std);

    const int B = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));

    auto grad_x = torch::empty_like(x);
    auto grad_gamma = torch::empty_like(gamma);
    auto grad_beta = torch::empty_like(gamma);

    zero_vector_kernel<<<(D + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        grad_gamma.data_ptr<float>(),
        grad_beta.data_ptr<float>(),
        D);

    layernorm_backward_x_kernel<<<B, kBlockSize>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        B,
        D);

    const dim3 param_block(kColTile);
    const dim3 param_grid((D + kColTile - 1) / kColTile, (B + kRowTile - 1) / kRowTile);
    layernorm_backward_param_kernel<<<param_grid, param_block>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        grad_beta.data_ptr<float>(),
        B,
        D);

    return {grad_x, grad_gamma, grad_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)");
    m.def("layernorm_backward", &layernorm_backward, "LayerNorm backward (CUDA)");
}
