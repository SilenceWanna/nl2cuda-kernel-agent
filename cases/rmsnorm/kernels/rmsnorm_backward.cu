#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

std::vector<torch::Tensor> rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    double eps
);

namespace {

constexpr int THREADS = 256;
constexpr int DGAMMA_COLS_PER_BLOCK = 256;
constexpr int DGAMMA_ROWS_PER_BLOCK = 8;

__inline__ __device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float block_sum(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warp_sum(v);
    if (lane == 0) {
        shared[warp] = v;
    }
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        out = (lane < ((blockDim.x + 31) >> 5)) ? shared[lane] : 0.0f;
        out = warp_sum(out);
    }
    return out;
}

__global__ void rmsnorm_backward_dx_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_x,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* gy_row = grad_y + static_cast<long long>(row) * cols;
    const float* x_row = x + static_cast<long long>(row) * cols;
    float* gx_row = grad_x + static_cast<long long>(row) * cols;

    float r = inv_rms[row];
    float dot = 0.0f;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        dot += gy_row[col] * gamma[col] * x_row[col];
    }

    float total_dot = block_sum(dot);
    __shared__ float shared_coeff;
    if (threadIdx.x == 0) {
        shared_coeff = (r * r * r) * (total_dot / static_cast<float>(cols));
    }
    __syncthreads();

    float coeff = shared_coeff;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float gy = gy_row[col];
        float xv = x_row[col];
        float g = gamma[col];
        gx_row[col] = gy * g * r - xv * coeff;
    }
}

__global__ void rmsnorm_backward_dgamma_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_gamma,
    int rows,
    int cols
) {
    int col = blockIdx.x * DGAMMA_COLS_PER_BLOCK + threadIdx.x;
    int row_start = blockIdx.y * DGAMMA_ROWS_PER_BLOCK;

    if (col >= cols) {
        return;
    }

    float sum = 0.0f;
    int row_end = row_start + DGAMMA_ROWS_PER_BLOCK;
    if (row_end > rows) {
        row_end = rows;
    }

    for (int row = row_start; row < row_end; ++row) {
        long long idx = static_cast<long long>(row) * cols + col;
        sum += grad_y[idx] * x[idx] * inv_rms[row];
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

    TORCH_CHECK(grad_y.size(0) == x.size(0) && grad_y.size(1) == x.size(1), "grad_y shape mismatch");
    TORCH_CHECK(x.size(1) == gamma.size(0), "gamma shape mismatch");
    TORCH_CHECK(x.size(0) == inv_rms.size(0), "inv_rms shape mismatch");

    TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(inv_rms.is_contiguous(), "inv_rms must be contiguous");

    const auto rows = static_cast<int>(x.size(0));
    const auto cols = static_cast<int>(x.size(1));

    auto grad_x = torch::empty_like(x);
    auto grad_gamma = torch::zeros_like(gamma);

    auto stream = at::cuda::getDefaultCUDAStream();

    rmsnorm_backward_dx_kernel<<<rows, THREADS, 0, stream>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        rows,
        cols
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 block(DGAMMA_COLS_PER_BLOCK);
    dim3 grid(
        (cols + DGAMMA_COLS_PER_BLOCK - 1) / DGAMMA_COLS_PER_BLOCK,
        (rows + DGAMMA_ROWS_PER_BLOCK - 1) / DGAMMA_ROWS_PER_BLOCK
    );
    rmsnorm_backward_dgamma_kernel<<<grid, block, 0, stream>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        rows,
        cols
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_gamma};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
    m.def("rmsnorm_backward", &rmsnorm_backward, "RMSNorm backward (CUDA)");
}
