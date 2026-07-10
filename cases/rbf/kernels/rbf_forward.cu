#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <vector>


namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_FLOAT32(x);  \
    CHECK_CONTIGUOUS(x)

__global__ void rbf_forward_vec4_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    float* __restrict__ K,
    int N,
    int M,
    int D,
    float gamma) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) {
        return;
    }

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(X + i * D);
    const float4* __restrict__ y4 = reinterpret_cast<const float4*>(Y + j * D);
    const int D4 = D >> 2;

    float dist = 0.0f;
#pragma unroll
    for (int q = 0; q < 16; ++q) {
        if (q < D4) {
            const float4 xv = x4[q];
            const float4 yv = y4[q];
            const float dx0 = xv.x - yv.x;
            const float dx1 = xv.y - yv.y;
            const float dx2 = xv.z - yv.z;
            const float dx3 = xv.w - yv.w;
            dist += dx0 * dx0 + dx1 * dx1 + dx2 * dx2 + dx3 * dx3;
        }
    }
    for (int q = 16; q < D4; ++q) {
        const float4 xv = x4[q];
        const float4 yv = y4[q];
        const float dx0 = xv.x - yv.x;
        const float dx1 = xv.y - yv.y;
        const float dx2 = xv.z - yv.z;
        const float dx3 = xv.w - yv.w;
        dist += dx0 * dx0 + dx1 * dx1 + dx2 * dx2 + dx3 * dx3;
    }

    K[i * M + j] = expf(-gamma * dist);
}

__global__ void rbf_forward_scalar_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    float* __restrict__ K,
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
    const float* __restrict__ xp = X + i * D;
    const float* __restrict__ yp = Y + j * D;
    for (int d = 0; d < D; ++d) {
        const float diff = xp[d] - yp[d];
        dist += diff * diff;
    }

    K[i * M + j] = expf(-gamma * dist);
}

}  // namespace


torch::Tensor rbf_forward(torch::Tensor X, torch::Tensor Y, double gamma) {
    CHECK_INPUT(X);
    CHECK_INPUT(Y);
    TORCH_CHECK(X.dim() == 2, "X must have shape [N, D]");
    TORCH_CHECK(Y.dim() == 2, "Y must have shape [M, D]");
    TORCH_CHECK(X.size(1) == Y.size(1), "X and Y must have the same D");

    const int N = static_cast<int>(X.size(0));
    const int D = static_cast<int>(X.size(1));
    const int M = static_cast<int>(Y.size(0));

    auto K = torch::empty({N, M}, X.options());

    constexpr int BX = 16;
    constexpr int BY = 16;
    const dim3 block(BX, BY);
    const dim3 grid((M + BX - 1) / BX, (N + BY - 1) / BY);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if ((D & 3) == 0) {
        rbf_forward_vec4_kernel<<<grid, block, 0, stream>>>(
            X.data_ptr<float>(),
            Y.data_ptr<float>(),
            K.data_ptr<float>(),
            N,
            M,
            D,
            static_cast<float>(gamma));
    } else {
        rbf_forward_scalar_kernel<<<grid, block, 0, stream>>>(
            X.data_ptr<float>(),
            Y.data_ptr<float>(),
            K.data_ptr<float>(),
            N,
            M,
            D,
            static_cast<float>(gamma));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return K;
}

std::vector<torch::Tensor> rbf_backward(
    torch::Tensor grad_out,
    torch::Tensor K,
    torch::Tensor X,
    torch::Tensor Y,
    double gamma);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbf_forward", &rbf_forward, "RBF Gaussian kernel matrix forward (CUDA)");
    m.def("rbf_backward", &rbf_backward, "RBF Gaussian kernel matrix backward (CUDA)");
}
