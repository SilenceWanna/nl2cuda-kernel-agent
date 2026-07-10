#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


std::vector<torch::Tensor> layernorm_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor mean,
    torch::Tensor rstd);


#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x); \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT(x)


__global__ void layernorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int B,
    int D,
    float eps) {
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= B) {
        return;
    }

    const int base = row * D;

    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        sum += x[base + d];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    const float m = smem[0] / static_cast<float>(D);

    float var_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        const float diff = x[base + d] - m;
        var_sum += diff * diff;
    }

    smem[tid] = var_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    const float var = smem[0] / static_cast<float>(D);
    const float inv_std = rsqrtf(var + eps);

    if (tid == 0) {
        mean[row] = m;
        rstd[row] = inv_std;
    }

    for (int d = tid; d < D; d += blockDim.x) {
        const float xhat = (x[base + d] - m) * inv_std;
        y[base + d] = xhat * gamma[d] + beta[d];
    }
}


std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);

    TORCH_CHECK(x.dim() == 2, "X must have shape [B, D]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
    TORCH_CHECK(beta.dim() == 1, "beta must have shape [D]");

    const int64_t B64 = x.size(0);
    const int64_t D64 = x.size(1);

    TORCH_CHECK(gamma.size(0) == D64, "gamma shape mismatch");
    TORCH_CHECK(beta.size(0) == D64, "beta shape mismatch");
    TORCH_CHECK(B64 <= static_cast<int64_t>(2147483647), "B is too large");
    TORCH_CHECK(D64 <= static_cast<int64_t>(2147483647), "D is too large");

    const int B = static_cast<int>(B64);
    const int D = static_cast<int>(D64);

    auto y = torch::empty_like(x);
    auto mean = torch::empty({B64}, x.options());
    auto rstd = torch::empty({B64}, x.options());

    const int threads = 256;
    const dim3 blocks(B);
    const size_t shared_bytes = threads * sizeof(float);

    layernorm_forward_kernel<<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        B,
        D,
        static_cast<float>(eps));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {y, mean, rstd};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward CUDA");
    m.def("backward", &layernorm_backward, "LayerNorm backward CUDA");
}
