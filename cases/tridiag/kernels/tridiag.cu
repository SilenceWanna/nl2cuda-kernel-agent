#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxN = 1024;

__global__ void tridiag_forward_kernel(
        const float* __restrict__ lower,
        const float* __restrict__ diag,
        const float* __restrict__ upper,
        const float* __restrict__ rhs,
        float* __restrict__ x,
        float* __restrict__ factors,
        int N) {
    extern __shared__ float shared[];
    float* shared_lower = shared;
    float* shared_diag = shared_lower + N;
    float* shared_upper = shared_diag + N;
    float* shared_rhs = shared_upper + N;

    const int64_t base = static_cast<int64_t>(blockIdx.x) * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        shared_lower[i] = lower[base + i];
        shared_diag[i] = diag[base + i];
        shared_upper[i] = upper[base + i];
        shared_rhs[i] = rhs[base + i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        shared_lower[0] = 0.0f;
        for (int i = 1; i < N; ++i) {
            const float multiplier = shared_lower[i] / shared_diag[i - 1];
            shared_lower[i] = multiplier;
            shared_diag[i] -= multiplier * shared_upper[i - 1];
            shared_rhs[i] -= multiplier * shared_rhs[i - 1];
        }

        shared_rhs[N - 1] /= shared_diag[N - 1];
        for (int i = N - 2; i >= 0; --i) {
            shared_rhs[i] =
                (shared_rhs[i] - shared_upper[i] * shared_rhs[i + 1]) /
                shared_diag[i];
        }
    }
    __syncthreads();

    const int64_t factor_stride = static_cast<int64_t>(gridDim.x) * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        x[base + i] = shared_rhs[i];
        factors[base + i] = shared_lower[i];
        factors[factor_stride + base + i] = shared_diag[i];
    }
}

__global__ void tridiag_backward_kernel(
        const float* __restrict__ grad_x,
        const float* __restrict__ upper,
        const float* __restrict__ x,
        const float* __restrict__ factors,
        float* __restrict__ gradients,
        int N) {
    extern __shared__ float shared[];
    float* shared_upper = shared;
    float* shared_x = shared_upper + N;
    float* shared_l = shared_x + N;
    float* shared_u = shared_l + N;
    float* shared_lambda = shared_u + N;

    const int64_t base = static_cast<int64_t>(blockIdx.x) * N;
    const int64_t tensor_stride = static_cast<int64_t>(gridDim.x) * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        shared_upper[i] = upper[base + i];
        shared_x[i] = x[base + i];
        shared_l[i] = factors[base + i];
        shared_u[i] = factors[tensor_stride + base + i];
        shared_lambda[i] = grad_x[base + i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        shared_lambda[0] /= shared_u[0];
        for (int i = 1; i < N; ++i) {
            shared_lambda[i] =
                (shared_lambda[i] -
                 shared_upper[i - 1] * shared_lambda[i - 1]) /
                shared_u[i];
        }
        for (int i = N - 2; i >= 0; --i) {
            shared_lambda[i] -= shared_l[i + 1] * shared_lambda[i + 1];
        }
    }
    __syncthreads();

    float* grad_lower = gradients;
    float* grad_diag = gradients + tensor_stride;
    float* grad_upper = gradients + 2 * tensor_stride;
    float* grad_rhs = gradients + 3 * tensor_stride;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float lambda = shared_lambda[i];
        grad_lower[base + i] =
            i == 0 ? 0.0f : -lambda * shared_x[i - 1];
        grad_diag[base + i] = -lambda * shared_x[i];
        grad_upper[base + i] =
            i + 1 == N ? 0.0f : -lambda * shared_x[i + 1];
        grad_rhs[base + i] = lambda;
    }
}

void validate_inputs(
        const torch::Tensor& lower,
        const torch::Tensor& diag,
        const torch::Tensor& upper,
        const torch::Tensor& rhs) {
    TORCH_CHECK(lower.is_cuda() && diag.is_cuda() &&
                    upper.is_cuda() && rhs.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(lower.scalar_type() == torch::kFloat32 &&
                    diag.scalar_type() == torch::kFloat32 &&
                    upper.scalar_type() == torch::kFloat32 &&
                    rhs.scalar_type() == torch::kFloat32,
                "all inputs must be float32");
    TORCH_CHECK(lower.is_contiguous() && diag.is_contiguous() &&
                    upper.is_contiguous() && rhs.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(lower.dim() == 2, "inputs must have shape [B, N]");
    TORCH_CHECK(diag.sizes() == lower.sizes() &&
                    upper.sizes() == lower.sizes() &&
                    rhs.sizes() == lower.sizes(),
                "all inputs must have the same shape");
    TORCH_CHECK(lower.size(0) > 0 && lower.size(1) > 0,
                "B and N must be positive");
    TORCH_CHECK(lower.size(1) <= kMaxN,
                "N must be at most ", kMaxN);
    TORCH_CHECK(lower.size(0) <= INT32_MAX, "B is too large");
}

void validate_backward(
        const torch::Tensor& grad_x,
        const torch::Tensor& upper,
        const torch::Tensor& x,
        const torch::Tensor& factors) {
    TORCH_CHECK(grad_x.is_cuda() && upper.is_cuda() &&
                    x.is_cuda() && factors.is_cuda(),
                "backward inputs must be CUDA tensors");
    TORCH_CHECK(grad_x.scalar_type() == torch::kFloat32 &&
                    upper.scalar_type() == torch::kFloat32 &&
                    x.scalar_type() == torch::kFloat32 &&
                    factors.scalar_type() == torch::kFloat32,
                "backward inputs must be float32");
    TORCH_CHECK(grad_x.is_contiguous() && upper.is_contiguous() &&
                    x.is_contiguous() && factors.is_contiguous(),
                "backward inputs must be contiguous");
    TORCH_CHECK(grad_x.dim() == 2 && upper.sizes() == grad_x.sizes() &&
                    x.sizes() == grad_x.sizes(),
                "grad_x, upper, and x must have shape [B, N]");
    TORCH_CHECK(factors.dim() == 3 && factors.size(0) == 2 &&
                    factors.size(1) == grad_x.size(0) &&
                    factors.size(2) == grad_x.size(1),
                "factors must have shape [2, B, N]");
    TORCH_CHECK(grad_x.size(1) <= kMaxN,
                "N must be at most ", kMaxN);
}

}  // namespace

std::vector<torch::Tensor> tridiag_forward(
        torch::Tensor lower,
        torch::Tensor diag,
        torch::Tensor upper,
        torch::Tensor rhs) {
    validate_inputs(lower, diag, upper, rhs);
    auto x = torch::empty_like(rhs);
    auto factors = torch::empty(
        {2, lower.size(0), lower.size(1)}, lower.options());

    const int B = static_cast<int>(lower.size(0));
    const int N = static_cast<int>(lower.size(1));
    const size_t shared_bytes = 4 * static_cast<size_t>(N) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tridiag_forward_kernel<<<B, kThreads, shared_bytes, stream>>>(
        lower.data_ptr<float>(), diag.data_ptr<float>(),
        upper.data_ptr<float>(), rhs.data_ptr<float>(),
        x.data_ptr<float>(), factors.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {x, factors};
}

std::vector<torch::Tensor> tridiag_backward(
        torch::Tensor grad_x,
        torch::Tensor upper,
        torch::Tensor x,
        torch::Tensor factors) {
    validate_backward(grad_x, upper, x, factors);
    auto gradients = torch::empty(
        {4, grad_x.size(0), grad_x.size(1)}, grad_x.options());

    const int B = static_cast<int>(grad_x.size(0));
    const int N = static_cast<int>(grad_x.size(1));
    const size_t shared_bytes = 5 * static_cast<size_t>(N) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tridiag_backward_kernel<<<B, kThreads, shared_bytes, stream>>>(
        grad_x.data_ptr<float>(), upper.data_ptr<float>(),
        x.data_ptr<float>(), factors.data_ptr<float>(),
        gradients.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        gradients.select(0, 0),
        gradients.select(0, 1),
        gradients.select(0, 2),
        gradients.select(0, 3),
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("tridiag_forward", &tridiag_forward,
               "Batched Thomas forward (CUDA)");
    module.def("tridiag_backward", &tridiag_backward,
               "Batched Thomas backward (CUDA)");
}
