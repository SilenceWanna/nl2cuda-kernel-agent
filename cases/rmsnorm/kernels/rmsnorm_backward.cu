#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int ROW_THREADS = 256;
constexpr int DGAMMA_COLS = 32;
constexpr int DGAMMA_ROWS = 16;
constexpr int DGAMMA_THREADS_Y = 8;

__global__ void rmsnorm_backward_x_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_x,
    int B,
    int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    const float* row_grad_y = grad_y + row * D;
    const float* row_x = x + row * D;

    float dot = 0.0f;
    for (int col = tid; col < D; col += blockDim.x) {
        dot += row_grad_y[col] * gamma[col] * row_x[col];
    }

    shared[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float row_dot = shared[0];
    float inv = inv_rms[row];
    float inv3_over_d = inv * inv * inv / static_cast<float>(D);

    float* row_grad_x = grad_x + row * D;
    for (int col = tid; col < D; col += blockDim.x) {
        float gy = row_grad_y[col];
        float xv = row_x[col];
        float gv = gamma[col];

        row_grad_x[col] = gy * gv * inv - xv * inv3_over_d * row_dot;
    }
}

__global__ void rmsnorm_backward_gamma_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_gamma,
    int B,
    int D) {
    int col = blockIdx.x * DGAMMA_COLS + threadIdx.x;
    int row_start = blockIdx.y * DGAMMA_ROWS;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float partial[DGAMMA_THREADS_Y][DGAMMA_COLS];

    float acc = 0.0f;

    if (col < D) {
        for (int r = row_start + ty; r < row_start + DGAMMA_ROWS && r < B; r += DGAMMA_THREADS_Y) {
            int idx = r * D + col;
            acc += grad_y[idx] * x[idx] * inv_rms[r];
        }
    }

    partial[ty][tx] = acc;
    __syncthreads();

    if (ty == 0 && col < D) {
        float sum = 0.0f;
        for (int k = 0; k < DGAMMA_THREADS_Y; ++k) {
            sum += partial[k][tx];
        }
        atomicAdd(grad_gamma + col, sum);
    }
}

}  // namespace

std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor inv_rms) {
    TORCH_CHECK(grad_y.is_cuda(), "grad_y must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(inv_rms.is_cuda(), "inv_rms must be a CUDA tensor");
    TORCH_CHECK(grad_y.dtype() == torch::kFloat32, "grad_y must be float32");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(inv_rms.dtype() == torch::kFloat32, "inv_rms must be float32");
    TORCH_CHECK(grad_y.dim() == 2, "grad_y must be 2D");
    TORCH_CHECK(x.dim() == 2, "X must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(inv_rms.dim() == 1, "inv_rms must be 1D");
    TORCH_CHECK(grad_y.size(0) == x.size(0), "grad_y.shape[0] must equal X.shape[0]");
    TORCH_CHECK(grad_y.size(1) == x.size(1), "grad_y.shape[1] must equal X.shape[1]");
    TORCH_CHECK(gamma.size(0) == x.size(1), "gamma.shape[0] must equal X.shape[1]");
    TORCH_CHECK(inv_rms.size(0) == x.size(0), "inv_rms.shape[0] must equal X.shape[0]");
    TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(inv_rms.is_contiguous(), "inv_rms must be contiguous");

    const int B = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));

    auto grad_x = torch::empty_like(x);
    auto grad_gamma = torch::zeros_like(gamma);

    const dim3 row_grid(B);
    const dim3 row_block(ROW_THREADS);
    const size_t row_shared_bytes = ROW_THREADS * sizeof(float);

    rmsnorm_backward_x_kernel<<<
        row_grid,
        row_block,
        row_shared_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        B,
        D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 gamma_grid(
        (D + DGAMMA_COLS - 1) / DGAMMA_COLS,
        (B + DGAMMA_ROWS - 1) / DGAMMA_ROWS);
    const dim3 gamma_block(DGAMMA_COLS, DGAMMA_THREADS_Y);

    rmsnorm_backward_gamma_kernel<<<
        gamma_grid,
        gamma_block,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        B,
        D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_gamma};
}
