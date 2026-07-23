// Fused tanh-GeGLU forward and backward for X[B, T, 2H].
// The last dimension is split into V and G. Forward returns V * GELU_tanh(G).
// Backward writes dV and dG directly into the corresponding halves of dX.

#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kCubicCoeff = 0.044715f;

__device__ __forceinline__ float gelu_tanh(float gate, float* tanh_u) {
    const float gate_sq = gate * gate;
    const float u = kSqrt2OverPi * (gate + kCubicCoeff * gate * gate_sq);
    const float t = tanhf(u);
    *tanh_u = t;
    return 0.5f * gate * (1.0f + t);
}

__device__ __forceinline__ float gelu_tanh_derivative(float gate, float tanh_u) {
    const float gate_sq = gate * gate;
    const float du_dgate = kSqrt2OverPi * (1.0f + 3.0f * kCubicCoeff * gate_sq);
    return 0.5f * (1.0f + tanh_u) +
           0.5f * gate * (1.0f - tanh_u * tanh_u) * du_dgate;
}

__global__ void geglu_forward_scalar_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int h) {
    const int row = static_cast<int>(blockIdx.y) +
                    static_cast<int>(gridDim.y) * static_cast<int>(blockIdx.z);
    if (row >= rows) {
        return;
    }

    const float* value = x + static_cast<int64_t>(row) * (2 * h);
    const float* gate = value + h;
    float* output = y + static_cast<int64_t>(row) * h;
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < h) {
        float t;
        output[j] = value[j] * gelu_tanh(gate[j], &t);
    }
}

__global__ void geglu_backward_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_y,
    float* __restrict__ grad_x,
    int rows,
    int h4) {
    const int row = static_cast<int>(blockIdx.y) +
                    static_cast<int>(gridDim.y) * static_cast<int>(blockIdx.z);
    if (row >= rows) {
        return;
    }

    const float4* value = reinterpret_cast<const float4*>(x) + static_cast<int64_t>(row) * (2 * h4);
    const float4* gate = value + h4;
    const float4* grad_output = reinterpret_cast<const float4*>(grad_y) + static_cast<int64_t>(row) * h4;
    float4* grad_value = reinterpret_cast<float4*>(grad_x) + static_cast<int64_t>(row) * (2 * h4);
    float4* grad_gate = grad_value + h4;

    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < h4) {
        const float4 v = value[j];
        const float4 g = gate[j];
        const float4 dy = grad_output[j];
        float t;
        float4 dv;
        float4 dg;

        const float gelu_x = gelu_tanh(g.x, &t);
        dv.x = dy.x * gelu_x;
        dg.x = dy.x * v.x * gelu_tanh_derivative(g.x, t);
        const float gelu_y = gelu_tanh(g.y, &t);
        dv.y = dy.y * gelu_y;
        dg.y = dy.y * v.y * gelu_tanh_derivative(g.y, t);
        const float gelu_z = gelu_tanh(g.z, &t);
        dv.z = dy.z * gelu_z;
        dg.z = dy.z * v.z * gelu_tanh_derivative(g.z, t);
        const float gelu_w = gelu_tanh(g.w, &t);
        dv.w = dy.w * gelu_w;
        dg.w = dy.w * v.w * gelu_tanh_derivative(g.w, t);

        grad_value[j] = dv;
        grad_gate[j] = dg;
    }
}

__global__ void geglu_backward_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_y,
    float* __restrict__ grad_x,
    int rows,
    int h) {
    const int row = static_cast<int>(blockIdx.y) +
                    static_cast<int>(gridDim.y) * static_cast<int>(blockIdx.z);
    if (row >= rows) {
        return;
    }

    const float* value = x + static_cast<int64_t>(row) * (2 * h);
    const float* gate = value + h;
    const float* grad_output = grad_y + static_cast<int64_t>(row) * h;
    float* grad_value = grad_x + static_cast<int64_t>(row) * (2 * h);
    float* grad_gate = grad_value + h;
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < h) {
        float t;
        const float gelu = gelu_tanh(gate[j], &t);
        grad_value[j] = grad_output[j] * gelu;
        grad_gate[j] = grad_output[j] * value[j] * gelu_tanh_derivative(gate[j], t);
    }
}

void validate_input(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(x.dim() == 3, "X must have shape [B, T, 2H]");
    TORCH_CHECK(x.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
                "all dimensions of X must be positive");
    TORCH_CHECK(x.size(2) % 2 == 0, "the last dimension of X must be even");
    TORCH_CHECK(x.size(0) <= INT_MAX / x.size(1), "B*T is too large");
    TORCH_CHECK(x.size(2) / 2 <= INT_MAX, "H is too large");
}

int rows_of(const torch::Tensor& x) {
    return static_cast<int>(x.size(0) * x.size(1));
}

int h_of(const torch::Tensor& x) {
    return static_cast<int>(x.size(2) / 2);
}

dim3 grid_for_rows(int rows, int chunks) {
    constexpr int kMaxGridY = 65535;
    const int rows_z = (rows + kMaxGridY - 1) / kMaxGridY;
    // Split rows evenly over y/z. Using y=65535 for a 65536-row tensor
    // would make nearly every block in z=1 empty.
    const int rows_y = (rows + rows_z - 1) / rows_z;
    return dim3(chunks, rows_y, rows_z);
}

}  // namespace

torch::Tensor geglu_forward(torch::Tensor x) {
    validate_input(x);
    const int rows = rows_of(x);
    const int h = h_of(x);
    auto y = torch::empty({x.size(0), x.size(1), h}, x.options());
    const auto stream = c10::cuda::getCurrentCUDAStream();

    const dim3 grid = grid_for_rows(rows, (h + kThreads - 1) / kThreads);
    geglu_forward_scalar_kernel<<<grid, kThreads, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), rows, h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor geglu_backward(torch::Tensor x, torch::Tensor grad_y) {
    validate_input(x);
    TORCH_CHECK(grad_y.is_cuda() && grad_y.scalar_type() == torch::kFloat32,
                "grad_y must be a CUDA float32 tensor");
    TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
    TORCH_CHECK(grad_y.dim() == 3 && grad_y.size(0) == x.size(0) &&
                    grad_y.size(1) == x.size(1) && grad_y.size(2) == x.size(2) / 2,
                "grad_y must have shape [B, T, H]");

    const int rows = rows_of(x);
    const int h = h_of(x);
    auto grad_x = torch::empty_like(x);
    const auto stream = c10::cuda::getCurrentCUDAStream();

    if ((h & 3) == 0) {
        const int h4 = h / 4;
        const dim3 grid = grid_for_rows(rows, (h4 + kThreads - 1) / kThreads);
        geglu_backward_vec4_kernel<<<grid, kThreads, 0, stream>>>(
            x.data_ptr<float>(), grad_y.data_ptr<float>(), grad_x.data_ptr<float>(), rows, h / 4);
    } else {
        const dim3 grid = grid_for_rows(rows, (h + kThreads - 1) / kThreads);
        geglu_backward_scalar_kernel<<<grid, kThreads, 0, stream>>>(
            x.data_ptr<float>(), grad_y.data_ptr<float>(), grad_x.data_ptr<float>(), rows, h);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("geglu_forward", &geglu_forward, "Tanh-GeGLU forward (CUDA)");
    module.def("geglu_backward", &geglu_backward, "Tanh-GeGLU backward (CUDA)");
}
