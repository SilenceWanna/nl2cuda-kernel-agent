#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


std::vector<torch::Tensor> rbf_backward(
    torch::Tensor grad_k,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor k,
    double gamma);


#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x); \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT(x)


__global__ void rbf_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ k,
    int N,
    int M,
    int D,
    float gamma) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= M) {
        return;
    }

    float dist = 0.0f;
    const int x_base = i * D;
    const int y_base = j * D;

    for (int d = 0; d < D; ++d) {
        const float diff = x[x_base + d] - y[y_base + d];
        dist += diff * diff;
    }

    k[i * M + j] = expf(-gamma * dist);
}


torch::Tensor rbf_forward(
    torch::Tensor x,
    torch::Tensor y,
    double gamma) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);

    TORCH_CHECK(x.dim() == 2, "X must have shape [N, D]");
    TORCH_CHECK(y.dim() == 2, "Y must have shape [M, D]");

    const int64_t N64 = x.size(0);
    const int64_t D64 = x.size(1);
    const int64_t M64 = y.size(0);

    TORCH_CHECK(y.size(1) == D64, "X and Y must have the same D");
    TORCH_CHECK(N64 <= static_cast<int64_t>(2147483647), "N is too large");
    TORCH_CHECK(M64 <= static_cast<int64_t>(2147483647), "M is too large");
    TORCH_CHECK(D64 <= static_cast<int64_t>(2147483647), "D is too large");

    const int N = static_cast<int>(N64);
    const int M = static_cast<int>(M64);
    const int D = static_cast<int>(D64);

    auto k = torch::empty({N64, M64}, x.options());

    const dim3 threads(16, 16);
    const dim3 blocks(
        (M + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y);

    rbf_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        k.data_ptr<float>(),
        N,
        M,
        D,
        static_cast<float>(gamma));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return k;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rbf_forward, "RBF forward CUDA");
    m.def("backward", &rbf_backward, "RBF backward CUDA");
}
