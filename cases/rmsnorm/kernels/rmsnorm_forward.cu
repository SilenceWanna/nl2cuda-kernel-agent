#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int THREADS = 256;

__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    float* __restrict__ inv_rms,
    int B,
    int D,
    float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    const float* row_x = x + row * D;
    float sum_sq = 0.0f;

    for (int col = tid; col < D; col += blockDim.x) {
        float v = row_x[col];
        sum_sq += v * v;
    }

    shared[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float inv = rsqrtf(shared[0] / static_cast<float>(D) + eps);

    if (tid == 0) {
        inv_rms[row] = inv;
    }

    float* row_y = y + row * D;
    for (int col = tid; col < D; col += blockDim.x) {
        row_y[col] = row_x[col] * inv * gamma[col];
    }
}

}  // namespace

std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor inv_rms);

std::vector<torch::Tensor> rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(x.dim() == 2, "X must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(x.size(1) == gamma.size(0), "X.shape[1] must equal gamma.shape[0]");
    TORCH_CHECK(x.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");

    const int B = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));

    auto y = torch::empty_like(x);
    auto inv_rms = torch::empty({B}, x.options());

    const dim3 grid(B);
    const dim3 block(THREADS);
    const size_t shared_bytes = THREADS * sizeof(float);

    rmsnorm_forward_kernel<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        y.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        B,
        D,
        static_cast<float>(eps));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {y, inv_rms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward");
    m.def("rmsnorm_backward", &rmsnorm_backward, "RMSNorm backward");
}
