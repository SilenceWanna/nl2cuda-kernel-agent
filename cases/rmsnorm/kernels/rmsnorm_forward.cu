#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

constexpr int THREADS = 256;

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

__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    float* __restrict__ inv_rms,
    int B,
    int D,
    float eps
) {
    int row = blockIdx.x;
    if (row >= B) {
        return;
    }

    const float* x_row = x + static_cast<long long>(row) * D;
    float* y_row = y + static_cast<long long>(row) * D;

    float sumsq = 0.0f;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        float v = x_row[col];
        sumsq += v * v;
    }

    sumsq = block_reduce_sum(sumsq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        float mean_sq = sumsq / static_cast<float>(D);
        s_inv_rms = rsqrtf(mean_sq + eps);
        inv_rms[row] = s_inv_rms;
    }
    __syncthreads();

    float row_inv_rms = s_inv_rms;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        y_row[col] = x_row[col] * row_inv_rms * gamma[col];
    }
}

}  // namespace

std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor gamma, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(x.size(1) == gamma.size(0), "x.size(1) must equal gamma.size(0)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");

    const auto B = static_cast<int>(x.size(0));
    const auto D = static_cast<int>(x.size(1));

    auto y = torch::empty_like(x);
    auto inv_rms = torch::empty({B}, x.options());

    const dim3 grid(B);
    const dim3 block(THREADS);
    auto stream = at::cuda::getDefaultCUDAStream();

    rmsnorm_forward_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        y.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        B,
        D,
        static_cast<float>(eps)
    );

    return {y, inv_rms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
}
