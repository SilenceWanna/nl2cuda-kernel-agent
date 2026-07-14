#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int DX_THREADS = 256;
constexpr int DG_COLS = 256;
constexpr int DG_ROWS = 8;

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void rmsnorm_backward_dx_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_x,
    int B,
    int D
) {
    int row = blockIdx.x;
    if (row >= B) {
        return;
    }

    const float* gy_row = grad_y + static_cast<long long>(row) * D;
    const float* x_row = x + static_cast<long long>(row) * D;
    float* gx_row = grad_x + static_cast<long long>(row) * D;

    float r = inv_rms[row];

    float dot = 0.0f;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        float x_hat = x_row[col] * r;
        float u = gy_row[col] * gamma[col];
        dot += u * x_hat;
    }

    dot = block_reduce_sum(dot);

    __shared__ float s_mean_dot;
    if (threadIdx.x == 0) {
        s_mean_dot = dot / static_cast<float>(D);
    }
    __syncthreads();

    float mean_dot = s_mean_dot;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        float x_hat = x_row[col] * r;
        float u = gy_row[col] * gamma[col];
        gx_row[col] = r * (u - x_hat * mean_dot);
    }
}

__global__ void rmsnorm_backward_dgamma_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_gamma,
    int B,
    int D
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start = blockIdx.y * DG_ROWS;

    if (col >= D) {
        return;
    }

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < DG_ROWS; ++i) {
        int row = row_start + i;
        if (row < B) {
            long long idx = static_cast<long long>(row) * D + col;
            float x_hat = x[idx] * inv_rms[row];
            sum += grad_y[idx] * x_hat;
        }
    }

    atomicAdd(grad_gamma + col, sum);
}

}  // namespace

std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor inv_rms
) {
    TORCH_CHECK(grad_y.is_cuda(), "grad_y must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(inv_rms.is_cuda(), "inv_rms must be a CUDA tensor");

    TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32, "grad_y must be float32");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(inv_rms.scalar_type() == torch::kFloat32, "inv_rms must be float32");

    TORCH_CHECK(grad_y.dim() == 2, "grad_y must be 2D");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(inv_rms.dim() == 1, "inv_rms must be 1D");

    TORCH_CHECK(grad_y.sizes() == x.sizes(), "grad_y and x must have same shape");
    TORCH_CHECK(x.size(1) == gamma.size(0), "x.size(1) must equal gamma.size(0)");
    TORCH_CHECK(x.size(0) == inv_rms.size(0), "x.size(0) must equal inv_rms.size(0)");

    TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(inv_rms.is_contiguous(), "inv_rms must be contiguous");

    const auto B = static_cast<int>(x.size(0));
    const auto D = static_cast<int>(x.size(1));

    auto grad_x = torch::empty_like(x);
    auto grad_gamma = torch::zeros_like(gamma);

    auto stream = at::cuda::getDefaultCUDAStream();

    rmsnorm_backward_dx_kernel<<<B, DX_THREADS, 0, stream>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        B,
        D
    );

    dim3 dg_block(DG_COLS);
    dim3 dg_grid((D + DG_COLS - 1) / DG_COLS, (B + DG_ROWS - 1) / DG_ROWS);

    rmsnorm_backward_dgamma_kernel<<<dg_grid, dg_block, 0, stream>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        B,
        D
    );

    return {grad_x, grad_gamma};
}
