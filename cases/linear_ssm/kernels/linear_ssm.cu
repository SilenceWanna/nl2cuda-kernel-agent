#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

__global__ void linear_ssm_forward_kernel(
        const float* __restrict__ x,
        float* __restrict__ y,
        int64_t sequences,
        int T,
        int C,
        float a,
        float b_coef) {
    int64_t sequence = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sequence >= sequences) {
        return;
    }

    const int c = static_cast<int>(sequence % C);
    const int64_t b = sequence / C;
    const int64_t base = b * static_cast<int64_t>(T) * C + c;
    float state = 0.0f;

    for (int t = 0; t < T; ++t) {
        const int64_t offset = base + static_cast<int64_t>(t) * C;
        state = a * state + b_coef * x[offset];
        y[offset] = state;
    }
}

__global__ void linear_ssm_backward_kernel(
        const float* __restrict__ grad_y,
        float* __restrict__ grad_x,
        int64_t sequences,
        int T,
        int C,
        float a,
        float b_coef) {
    int64_t sequence = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sequence >= sequences) {
        return;
    }

    const int c = static_cast<int>(sequence % C);
    const int64_t b = sequence / C;
    const int64_t base = b * static_cast<int64_t>(T) * C + c;
    float adjoint = 0.0f;

    for (int t = T - 1; t >= 0; --t) {
        const int64_t offset = base + static_cast<int64_t>(t) * C;
        adjoint = grad_y[offset] + a * adjoint;
        grad_x[offset] = b_coef * adjoint;
    }
}

void validate_input(const torch::Tensor& input, const char* name) {
    TORCH_CHECK(input.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(input.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(input.dim() == 3, name, " must have shape [B, T, C]");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 && input.size(2) > 0,
                name, " dimensions must be positive");
    TORCH_CHECK(input.size(1) <= INT32_MAX, "T is too large");
    TORCH_CHECK(input.size(2) <= INT32_MAX, "C is too large");
}

void validate_params(double a, double b_coef) {
    TORCH_CHECK(std::isfinite(a), "a must be finite");
    TORCH_CHECK(std::isfinite(b_coef), "b_coef must be finite");
}

}  // namespace

torch::Tensor linear_ssm_forward(torch::Tensor x, double a, double b_coef) {
    validate_input(x, "X");
    validate_params(a, b_coef);
    auto y = torch::empty_like(x);

    const int T = static_cast<int>(x.size(1));
    const int C = static_cast<int>(x.size(2));
    const int64_t sequences = x.size(0) * x.size(2);
    const int blocks = static_cast<int>((sequences + kThreads - 1) / kThreads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    linear_ssm_forward_kernel<<<blocks, kThreads, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), sequences, T, C,
        static_cast<float>(a), static_cast<float>(b_coef));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor linear_ssm_backward(torch::Tensor grad_y, double a, double b_coef) {
    validate_input(grad_y, "grad_y");
    validate_params(a, b_coef);
    auto grad_x = torch::empty_like(grad_y);

    const int T = static_cast<int>(grad_y.size(1));
    const int C = static_cast<int>(grad_y.size(2));
    const int64_t sequences = grad_y.size(0) * grad_y.size(2);
    const int blocks = static_cast<int>((sequences + kThreads - 1) / kThreads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    linear_ssm_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        grad_y.data_ptr<float>(), grad_x.data_ptr<float>(), sequences, T, C,
        static_cast<float>(a), static_cast<float>(b_coef));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("linear_ssm_forward", &linear_ssm_forward, "Linear SSM forward (CUDA)");
    module.def("linear_ssm_backward", &linear_ssm_backward, "Linear SSM backward (CUDA)");
}
