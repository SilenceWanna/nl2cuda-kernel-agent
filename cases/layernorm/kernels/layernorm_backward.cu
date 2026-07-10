#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int THREADS = 256;
constexpr int PARAM_COL_TILE = 256;
constexpr int PARAM_ROW_TILE = 32;

__global__ void layernorm_backward_x_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_x,
    int b,
    int d) {
    __shared__ float shared_sum1[THREADS];
    __shared__ float shared_sum2[THREADS];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= b) {
        return;
    }

    const float* x_row = x + static_cast<long long>(row) * d;
    const float* go_row = grad_out + static_cast<long long>(row) * d;
    float row_mean = mean[row];
    float row_rstd = rstd[row];

    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int col = tid; col < d; col += THREADS) {
        float xhat = (x_row[col] - row_mean) * row_rstd;
        float dxhat = go_row[col] * gamma[col];
        sum1 += dxhat;
        sum2 += dxhat * xhat;
    }

    shared_sum1[tid] = sum1;
    shared_sum2[tid] = sum2;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum1[tid] += shared_sum1[tid + stride];
            shared_sum2[tid] += shared_sum2[tid + stride];
        }
        __syncthreads();
    }

    float inv_d = 1.0f / static_cast<float>(d);
    float row_sum1 = shared_sum1[0];
    float row_sum2 = shared_sum2[0];
    float* gx_row = grad_x + static_cast<long long>(row) * d;

    for (int col = tid; col < d; col += THREADS) {
        float xhat = (x_row[col] - row_mean) * row_rstd;
        float dxhat = go_row[col] * gamma[col];
        gx_row[col] = (dxhat - row_sum1 * inv_d - xhat * row_sum2 * inv_d) * row_rstd;
    }
}

__global__ void layernorm_backward_param_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int b,
    int d) {
    int col = blockIdx.x * PARAM_COL_TILE + threadIdx.x;
    if (col >= d) {
        return;
    }

    int row_begin = blockIdx.y * PARAM_ROW_TILE;
    int row_end = row_begin + PARAM_ROW_TILE < b ? row_begin + PARAM_ROW_TILE : b;

    float gamma_acc = 0.0f;
    float beta_acc = 0.0f;
    for (int row = row_begin; row < row_end; ++row) {
        long long offset = static_cast<long long>(row) * d + col;
        float go = grad_out[offset];
        float xhat = (x[offset] - mean[row]) * rstd[row];
        gamma_acc += go * xhat;
        beta_acc += go;
    }

    atomicAdd(grad_gamma + col, gamma_acc);
    atomicAdd(grad_beta + col, beta_acc);
}

}  // namespace

std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps);

std::vector<torch::Tensor> layernorm_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor mean,
    torch::Tensor rstd) {
    TORCH_CHECK(grad_out.is_cuda() && x.is_cuda() && gamma.is_cuda() &&
                mean.is_cuda() && rstd.is_cuda(),
                "LayerNorm backward tensors must be CUDA tensors");
    TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 &&
                x.scalar_type() == torch::kFloat32 &&
                gamma.scalar_type() == torch::kFloat32 &&
                mean.scalar_type() == torch::kFloat32 &&
                rstd.scalar_type() == torch::kFloat32,
                "LayerNorm backward only supports float32");
    TORCH_CHECK(grad_out.is_contiguous() && x.is_contiguous() &&
                gamma.is_contiguous() && mean.is_contiguous() && rstd.is_contiguous(),
                "LayerNorm backward tensors must be contiguous");
    TORCH_CHECK(x.dim() == 2 && grad_out.dim() == 2, "LayerNorm X/grad_out must be 2D");

    int b = static_cast<int>(x.size(0));
    int d = static_cast<int>(x.size(1));
    TORCH_CHECK(grad_out.size(0) == b && grad_out.size(1) == d,
                "LayerNorm grad_out shape mismatch");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == d,
                "LayerNorm gamma shape mismatch");
    TORCH_CHECK(mean.dim() == 1 && rstd.dim() == 1 &&
                mean.size(0) == b && rstd.size(0) == b,
                "LayerNorm saved statistics shape mismatch");

    auto grad_x = torch::empty_like(x);
    auto grad_gamma = torch::empty_like(gamma);
    auto grad_beta = torch::empty_like(gamma);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        grad_gamma.data_ptr<float>(), 0, d * sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        grad_beta.data_ptr<float>(), 0, d * sizeof(float), stream));

    layernorm_backward_x_kernel<<<b, THREADS, 0, stream>>>(
        grad_out.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        b,
        d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 param_block(PARAM_COL_TILE);
    dim3 param_grid(
        (d + PARAM_COL_TILE - 1) / PARAM_COL_TILE,
        (b + PARAM_ROW_TILE - 1) / PARAM_ROW_TILE);
    layernorm_backward_param_kernel<<<param_grid, param_block, 0, stream>>>(
        grad_out.data_ptr<float>(),
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        grad_beta.data_ptr<float>(),
        b,
        d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_gamma, grad_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)");
    m.def("layernorm_backward", &layernorm_backward, "LayerNorm backward (CUDA)");
}
