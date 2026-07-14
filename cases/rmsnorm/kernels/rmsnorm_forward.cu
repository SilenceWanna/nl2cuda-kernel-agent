#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

constexpr int THREADS = 256;

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

__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    float* __restrict__ inv_rms,
    int rows,
    int cols,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* x_row = x + static_cast<long long>(row) * cols;
    float* y_row = y + static_cast<long long>(row) * cols;

    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float xv = x_row[col];
        sum_sq += xv * xv;
    }

    float total_sum_sq = block_sum(sum_sq);
    __shared__ float shared_inv_rms;
    if (threadIdx.x == 0) {
        float mean_sq = total_sum_sq / static_cast<float>(cols);
        shared_inv_rms = rsqrtf(mean_sq + eps);
        inv_rms[row] = shared_inv_rms;
    }
    __syncthreads();

    float row_inv_rms = shared_inv_rms;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        y_row[col] = x_row[col] * row_inv_rms * gamma[col];
    }
}

}  // namespace

std::vector<torch::Tensor> rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(x.size(1) == gamma.size(0), "gamma shape mismatch");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");

    const auto rows = static_cast<int>(x.size(0));
    const auto cols = static_cast<int>(x.size(1));

    auto y = torch::empty_like(x);
    auto inv_rms = torch::empty({rows}, x.options());

    const dim3 grid(rows);
    const dim3 block(THREADS);
    auto stream = at::cuda::getDefaultCUDAStream();

    rmsnorm_forward_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        y.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        rows,
        cols,
        static_cast<float>(eps)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {y, inv_rms};
}
