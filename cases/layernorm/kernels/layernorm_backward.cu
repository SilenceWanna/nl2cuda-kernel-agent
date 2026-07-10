#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x); \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT(x)


__global__ void layernorm_dx_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_x,
    int B,
    int D) {
    extern __shared__ float smem[];

    float* smem_sum1 = smem;
    float* smem_sum2 = smem + blockDim.x;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= B) {
        return;
    }

    const int base = row * D;
    const float m = mean[row];
    const float inv_std = rstd[row];

    float local_sum1 = 0.0f;
    float local_sum2 = 0.0f;

    for (int d = tid; d < D; d += blockDim.x) {
        const float xhat = (x[base + d] - m) * inv_std;
        const float dy_gamma = grad_y[base + d] * gamma[d];

        local_sum1 += dy_gamma;
        local_sum2 += dy_gamma * xhat;
    }

    smem_sum1[tid] = local_sum1;
    smem_sum2[tid] = local_sum2;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_sum1[tid] += smem_sum1[tid + stride];
            smem_sum2[tid] += smem_sum2[tid + stride];
        }
        __syncthreads();
    }

    const float sum1 = smem_sum1[0];
    const float sum2 = smem_sum2[0];
    const float inv_D = 1.0f / static_cast<float>(D);

    for (int d = tid; d < D; d += blockDim.x) {
        const float xhat = (x[base + d] - m) * inv_std;
        const float dy_gamma = grad_y[base + d] * gamma[d];

        grad_x[base + d] = inv_std * (dy_gamma - sum1 * inv_D - xhat * sum2 * inv_D);
    }
}


__global__ void layernorm_param_grads_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int B,
    int D) {
    extern __shared__ float smem[];

    float* smem_gamma = smem;
    float* smem_beta = smem + blockDim.x;

    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= D) {
        return;
    }

    float local_gamma = 0.0f;
    float local_beta = 0.0f;

    for (int row = tid; row < B; row += blockDim.x) {
        const int idx = row * D + col;
        const float gy = grad_y[idx];
        const float xhat = (x[idx] - mean[row]) * rstd[row];

        local_gamma += gy * xhat;
        local_beta += gy;
    }

    smem_gamma[tid] = local_gamma;
    smem_beta[tid] = local_beta;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_gamma[tid] += smem_gamma[tid + stride];
            smem_beta[tid] += smem_beta[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_gamma[col] = smem_gamma[0];
        grad_beta[col] = smem_beta[0];
    }
}


std::vector<torch::Tensor> layernorm_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor mean,
    torch::Tensor rstd) {
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);
    CHECK_INPUT(mean);
    CHECK_INPUT(rstd);

    TORCH_CHECK(grad_y.dim() == 2, "grad_y must have shape [B, D]");
    TORCH_CHECK(x.dim() == 2, "X must have shape [B, D]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
    TORCH_CHECK(mean.dim() == 1, "mean must have shape [B]");
    TORCH_CHECK(rstd.dim() == 1, "rstd must have shape [B]");

    const int64_t B64 = x.size(0);
    const int64_t D64 = x.size(1);

    TORCH_CHECK(grad_y.size(0) == B64, "grad_y B mismatch");
    TORCH_CHECK(grad_y.size(1) == D64, "grad_y D mismatch");
    TORCH_CHECK(gamma.size(0) == D64, "gamma shape mismatch");
    TORCH_CHECK(mean.size(0) == B64, "mean shape mismatch");
    TORCH_CHECK(rstd.size(0) == B64, "rstd shape mismatch");
    TORCH_CHECK(B64 <= static_cast<int64_t>(2147483647), "B is too large");
    TORCH_CHECK(D64 <= static_cast<int64_t>(2147483647), "D is too large");

    const int B = static_cast<int>(B64);
    const int D = static_cast<int>(D64);

    auto grad_x = torch::empty_like(x);
    auto grad_gamma = torch::empty_like(gamma);
    auto grad_beta = torch::empty_like(gamma);

    const int threads = 256;

    const dim3 dx_blocks(B);
    const size_t dx_shared_bytes = 2 * threads * sizeof(float);

    layernorm_dx_kernel<<<dx_blocks, threads, dx_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        B,
        D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 param_blocks(D);
    const size_t param_shared_bytes = 2 * threads * sizeof(float);

    layernorm_param_grads_kernel<<<param_blocks, threads, param_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        grad_beta.data_ptr<float>(),
        B,
        D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_gamma, grad_beta};
}
