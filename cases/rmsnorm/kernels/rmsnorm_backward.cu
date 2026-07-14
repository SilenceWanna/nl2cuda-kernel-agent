#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

std::vector<torch::Tensor> rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
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
    const int num_warps = (blockDim.x + 31) >> 5;

    v = warp_reduce_sum(v);

    if (lane == 0) {
        shared[wid] = v;
    }
    __syncthreads();

    v = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) {
        v = warp_reduce_sum(v);
    }

    if (threadIdx.x == 0) {
        shared[0] = v;
    }
    __syncthreads();

    return shared[0];
}

__global__ void rmsnorm_backward_x_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_x,
    int B,
    int D) {
    const int b = blockIdx.x;
    if (b >= B) {
        return;
    }

    const int row = b * D;
    const float inv = inv_rms[b];
    const float inv3_over_d = (inv * inv * inv) / static_cast<float>(D);

    float sum_vx = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = grad_y[row + d] * gamma[d];
        sum_vx += v * x[row + d];
    }

    const float total_vx = block_reduce_sum(sum_vx);

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float gy = grad_y[row + d];
        const float xv = x[row + d];
        const float g = gamma[d];
        grad_x[row + d] = gy * g * inv - xv * inv3_over_d * total_vx;
    }
}

__global__ void zero_vector_kernel(float* __restrict__ a, int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        a[i] = 0.0f;
    }
}

__global__ void rmsnorm_backward_gamma_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_gamma,
    int B,
    int D) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_start = blockIdx.y * kRowTile;

    if (col >= D) {
        return;
    }

    float sum_gamma = 0.0f;

#pragma unroll
    for (int r = 0; r < kRowTile; ++r) {
        const int b = row_start + r;
        if (b < B) {
            const int idx = b * D + col;
            sum_gamma += grad_y[idx] * (x[idx] * inv_rms[b]);
        }
    }

    atomicAdd(grad_gamma + col, sum_gamma);
}

void check_backward_inputs(
    const torch::Tensor& grad_y,
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& inv_rms) {
    TORCH_CHECK(grad_y.is_cuda(), "grad_y must be CUDA");
    TORCH_CHECK(x.is_cuda(), "X must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(inv_rms.is_cuda(), "inv_rms must be CUDA");
    TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32, "grad_y must be float32");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(inv_rms.scalar_type() == torch::kFloat32, "inv_rms must be float32");
    TORCH_CHECK(grad_y.dim() == 2, "grad_y must have shape [B, D]");
    TORCH_CHECK(x.dim() == 2, "X must have shape [B, D]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
    TORCH_CHECK(inv_rms.dim() == 1, "inv_rms must have shape [B]");
    TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(inv_rms.is_contiguous(), "inv_rms must be contiguous");
    TORCH_CHECK(grad_y.size(0) == x.size(0), "grad_y B must match X B");
    TORCH_CHECK(grad_y.size(1) == x.size(1), "grad_y D must match X D");
    TORCH_CHECK(gamma.size(0) == x.size(1), "gamma length must equal D");
    TORCH_CHECK(inv_rms.size(0) == x.size(0), "inv_rms length must equal B");
}

}  // namespace

std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor inv_rms) {
    check_backward_inputs(grad_y, x, gamma, inv_rms);

    const int B = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));

    auto grad_x = torch::empty_like(x);
    auto grad_gamma = torch::empty_like(gamma);

    zero_vector_kernel<<<(D + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        grad_gamma.data_ptr<float>(),
        D);

    rmsnorm_backward_x_kernel<<<B, kBlockSize>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        B,
        D);

    const dim3 gamma_block(kColTile);
    const dim3 gamma_grid((D + kColTile - 1) / kColTile, (B + kRowTile - 1) / kRowTile);
    rmsnorm_backward_gamma_kernel<<<gamma_grid, gamma_block>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        B,
        D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_gamma};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
    m.def("rmsnorm_backward", &rmsnorm_backward, "RMSNorm backward (CUDA)");
}
