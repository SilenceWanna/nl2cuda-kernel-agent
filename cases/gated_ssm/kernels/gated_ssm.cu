#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__global__ void gated_ssm_forward_kernel(
        const float* __restrict__ x,
        const float* __restrict__ w,
        const float* __restrict__ bias,
        float* __restrict__ y,
        int64_t sequences,
        int T,
        int C) {
    const int64_t sequence =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sequence >= sequences) {
        return;
    }

    const int c = static_cast<int>(sequence % C);
    const int64_t batch = sequence / C;
    const int64_t base = batch * static_cast<int64_t>(T) * C + c;
    const float wc = w[c];
    const float bc = bias[c];
    float state = 0.0f;

    for (int t = 0; t < T; ++t) {
        const int64_t offset = base + static_cast<int64_t>(t) * C;
        const float xt = x[offset];
        const float z = sigmoid(wc * xt + bc);
        state = z * state + (1.0f - z) * xt;
        y[offset] = state;
    }
}

__global__ void gated_ssm_backward_kernel(
        const float* __restrict__ grad_y,
        const float* __restrict__ x,
        const float* __restrict__ w,
        const float* __restrict__ bias,
        const float* __restrict__ y,
        float* __restrict__ grad_x,
        float* __restrict__ grad_w,
        float* __restrict__ grad_b,
        int64_t sequences,
        int T,
        int C) {
    const int64_t sequence =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sequence >= sequences) {
        return;
    }

    const int c = static_cast<int>(sequence % C);
    const int64_t batch = sequence / C;
    const int64_t base = batch * static_cast<int64_t>(T) * C + c;
    const float wc = w[c];
    const float bc = bias[c];
    float state_adjoint = 0.0f;
    float dw = 0.0f;
    float db = 0.0f;

    for (int t = T - 1; t >= 0; --t) {
        const int64_t offset = base + static_cast<int64_t>(t) * C;
        const float xt = x[offset];
        const float previous_state =
            t == 0 ? 0.0f : y[offset - static_cast<int64_t>(C)];
        const float z = sigmoid(wc * xt + bc);
        state_adjoint += grad_y[offset];

        const float dz = state_adjoint * (previous_state - xt);
        const float dlogit = dz * z * (1.0f - z);
        grad_x[offset] = state_adjoint * (1.0f - z) + dlogit * wc;
        dw += dlogit * xt;
        db += dlogit;
        state_adjoint *= z;
    }

    atomicAdd(grad_w + c, dw);
    atomicAdd(grad_b + c, db);
}

void validate_inputs(
        const torch::Tensor& x,
        const torch::Tensor& w,
        const torch::Tensor& bias) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && bias.is_cuda(),
                "X, w, and b must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                    w.scalar_type() == torch::kFloat32 &&
                    bias.scalar_type() == torch::kFloat32,
                "X, w, and b must be float32");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && bias.is_contiguous(),
                "X, w, and b must be contiguous");
    TORCH_CHECK(x.dim() == 3, "X must have shape [B, T, C]");
    TORCH_CHECK(w.dim() == 1 && bias.dim() == 1,
                "w and b must have shape [C]");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
                "X dimensions must be positive");
    TORCH_CHECK(w.size(0) == x.size(2) && bias.size(0) == x.size(2),
                "w and b must match X's channel dimension");
    TORCH_CHECK(x.size(1) <= INT32_MAX && x.size(2) <= INT32_MAX,
                "T and C must fit in int32");
}

void validate_backward(
        const torch::Tensor& grad_y,
        const torch::Tensor& x,
        const torch::Tensor& w,
        const torch::Tensor& bias,
        const torch::Tensor& y) {
    validate_inputs(x, w, bias);
    TORCH_CHECK(grad_y.is_cuda() && y.is_cuda(),
                "grad_y and Y must be CUDA tensors");
    TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32 &&
                    y.scalar_type() == torch::kFloat32,
                "grad_y and Y must be float32");
    TORCH_CHECK(grad_y.is_contiguous() && y.is_contiguous(),
                "grad_y and Y must be contiguous");
    TORCH_CHECK(grad_y.sizes() == x.sizes() && y.sizes() == x.sizes(),
                "grad_y and Y must match X's shape");
}

}  // namespace

torch::Tensor gated_ssm_forward(
        torch::Tensor x,
        torch::Tensor w,
        torch::Tensor bias) {
    validate_inputs(x, w, bias);
    auto y = torch::empty_like(x);

    const int T = static_cast<int>(x.size(1));
    const int C = static_cast<int>(x.size(2));
    const int64_t sequences = x.size(0) * x.size(2);
    const int blocks = static_cast<int>((sequences + kThreads - 1) / kThreads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    gated_ssm_forward_kernel<<<blocks, kThreads, 0, stream>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), bias.data_ptr<float>(),
        y.data_ptr<float>(), sequences, T, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

std::vector<torch::Tensor> gated_ssm_backward(
        torch::Tensor grad_y,
        torch::Tensor x,
        torch::Tensor w,
        torch::Tensor bias,
        torch::Tensor y) {
    validate_backward(grad_y, x, w, bias, y);
    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::zeros_like(w);
    auto grad_b = torch::zeros_like(bias);

    const int T = static_cast<int>(x.size(1));
    const int C = static_cast<int>(x.size(2));
    const int64_t sequences = x.size(0) * x.size(2);
    const int blocks = static_cast<int>((sequences + kThreads - 1) / kThreads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    gated_ssm_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        grad_y.data_ptr<float>(), x.data_ptr<float>(), w.data_ptr<float>(),
        bias.data_ptr<float>(), y.data_ptr<float>(), grad_x.data_ptr<float>(),
        grad_w.data_ptr<float>(), grad_b.data_ptr<float>(), sequences, T, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_x, grad_w, grad_b};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("gated_ssm_forward", &gated_ssm_forward,
               "Gated SSM forward (CUDA)");
    module.def("gated_ssm_backward", &gated_ssm_backward,
               "Gated SSM backward (CUDA)");
}
