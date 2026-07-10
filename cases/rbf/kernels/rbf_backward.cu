#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;

__global__ void rbf_backward_x_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ out,
    float* __restrict__ grad_x,
    int n,
    int m,
    int d,
    float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * d;
    if (idx >= total) {
        return;
    }

    int dim = idx % d;
    int i = idx / d;
    float x_val = x[static_cast<long long>(i) * d + dim];
    const float* grad_row = grad_out + static_cast<long long>(i) * m;
    const float* out_row = out + static_cast<long long>(i) * m;
    float acc = 0.0f;

    for (int j = 0; j < m; ++j) {
        float weighted_k = grad_row[j] * out_row[j];
        float y_val = y[static_cast<long long>(j) * d + dim];
        acc += weighted_k * (y_val - x_val);
    }
    grad_x[idx] = scale * acc;
}

__global__ void rbf_backward_y_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ out,
    float* __restrict__ grad_y,
    int n,
    int m,
    int d,
    float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * d;
    if (idx >= total) {
        return;
    }

    int dim = idx % d;
    int j = idx / d;
    float y_val = y[static_cast<long long>(j) * d + dim];
    float acc = 0.0f;

    for (int i = 0; i < n; ++i) {
        long long offset = static_cast<long long>(i) * m + j;
        float weighted_k = grad_out[offset] * out[offset];
        float x_val = x[static_cast<long long>(i) * d + dim];
        acc += weighted_k * (x_val - y_val);
    }
    grad_y[idx] = scale * acc;
}

}  // namespace

torch::Tensor rbf_forward(torch::Tensor x, torch::Tensor y, double gamma);

std::vector<torch::Tensor> rbf_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor out,
    double gamma) {
    TORCH_CHECK(grad_out.is_cuda() && x.is_cuda() && y.is_cuda() && out.is_cuda(),
                "RBF backward tensors must be CUDA tensors");
    TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 &&
                x.scalar_type() == torch::kFloat32 &&
                y.scalar_type() == torch::kFloat32 &&
                out.scalar_type() == torch::kFloat32,
                "RBF backward only supports float32");
    TORCH_CHECK(grad_out.is_contiguous() && x.is_contiguous() && y.is_contiguous() && out.is_contiguous(),
                "RBF backward tensors must be contiguous");

    int n = static_cast<int>(x.size(0));
    int d = static_cast<int>(x.size(1));
    int m = static_cast<int>(y.size(0));
    TORCH_CHECK(y.size(1) == d, "RBF backward feature dimensions must match");
    TORCH_CHECK(grad_out.size(0) == n && grad_out.size(1) == m, "RBF grad_out shape mismatch");
    TORCH_CHECK(out.size(0) == n && out.size(1) == m, "RBF saved output shape mismatch");

    auto grad_x = torch::empty_like(x);
    auto grad_y = torch::empty_like(y);
    float scale = 2.0f * static_cast<float>(gamma);

    int gx_total = n * d;
    int gy_total = m * d;
    int gx_blocks = (gx_total + THREADS - 1) / THREADS;
    int gy_blocks = (gy_total + THREADS - 1) / THREADS;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rbf_backward_x_kernel<<<gx_blocks, THREADS, 0, stream>>>(
        grad_out.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        n,
        m,
        d,
        scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    rbf_backward_y_kernel<<<gy_blocks, THREADS, 0, stream>>>(
        grad_out.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        n,
        m,
        d,
        scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbf_forward", &rbf_forward, "RBF forward (CUDA)");
    m.def("rbf_backward", &rbf_backward, "RBF backward (CUDA)");
}
