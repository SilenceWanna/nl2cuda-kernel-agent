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

constexpr int D64 = 64;
constexpr int THREADS_D64 = 256;
constexpr int GROUPS_D64 = THREADS_D64 / D64;

__global__ __launch_bounds__(THREADS_D64, 2) void rbf_backward_dx_d64_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ K,
    const float* __restrict__ X,
    const float* __restrict__ Y,
    float* __restrict__ dX,
    int N,
    int M,
    float scale) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int d = tid & (D64 - 1);
    const int lane_group = tid >> 6;

    __shared__ float partial[GROUPS_D64][D64];

    const float xv = X[i * D64 + d];
    float acc = 0.0f;

    for (int j = lane_group; j < M; j += GROUPS_D64) {
        const float coeff = scale * grad_out[i * M + j] * K[i * M + j];
        acc += coeff * (xv - Y[j * D64 + d]);
    }

    partial[lane_group][d] = acc;
    __syncthreads();

    if (lane_group == 0) {
        float sum = partial[0][d];
#pragma unroll
        for (int g = 1; g < GROUPS_D64; ++g) {
            sum += partial[g][d];
        }
        dX[i * D64 + d] = sum;
    }
}

__global__ __launch_bounds__(THREADS_D64, 2) void rbf_backward_dy_d64_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ K,
    const float* __restrict__ X,
    const float* __restrict__ Y,
    float* __restrict__ dY,
    int N,
    int M,
    float scale) {
    const int j = blockIdx.x;
    const int tid = threadIdx.x;
    const int d = tid & (D64 - 1);
    const int lane_group = tid >> 6;

    __shared__ float partial[GROUPS_D64][D64];

    const float yv = Y[j * D64 + d];
    float acc = 0.0f;

    for (int i = lane_group; i < N; i += GROUPS_D64) {
        const float coeff = -scale * grad_out[i * M + j] * K[i * M + j];
        acc += coeff * (X[i * D64 + d] - yv);
    }

    partial[lane_group][d] = acc;
    __syncthreads();

    if (lane_group == 0) {
        float sum = partial[0][d];
#pragma unroll
        for (int g = 1; g < GROUPS_D64; ++g) {
            sum += partial[g][d];
        }
        dY[j * D64 + d] = sum;
    }
}

__global__ void rbf_backward_dx_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ K,
    const float* __restrict__ X,
    const float* __restrict__ Y,
    float* __restrict__ dX,
    int N,
    int M,
    int D,
    float gamma) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * D;
    if (idx >= total) {
        return;
    }

    const int i = idx / D;
    const int d = idx - i * D;
    const float xv = X[idx];

    float acc = 0.0f;
    for (int j = 0; j < M; ++j) {
        const float coeff = -2.0f * gamma * grad_out[i * M + j] * K[i * M + j];
        acc += coeff * (xv - Y[j * D + d]);
    }
    dX[idx] = acc;
}

__global__ void rbf_backward_dy_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ K,
    const float* __restrict__ X,
    const float* __restrict__ Y,
    float* __restrict__ dY,
    int N,
    int M,
    int D,
    float gamma) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * D;
    if (idx >= total) {
        return;
    }

    const int j = idx / D;
    const int d = idx - j * D;
    const float yv = Y[idx];

    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float coeff = 2.0f * gamma * grad_out[i * M + j] * K[i * M + j];
        acc += coeff * (X[i * D + d] - yv);
    }
    dY[idx] = acc;
}

}  // namespace


std::vector<torch::Tensor> rbf_backward(
    torch::Tensor grad_out,
    torch::Tensor K,
    torch::Tensor X,
    torch::Tensor Y,
    double gamma) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(K);
    CHECK_INPUT(X);
    CHECK_INPUT(Y);
    TORCH_CHECK(X.dim() == 2, "X must have shape [N, D]");
    TORCH_CHECK(Y.dim() == 2, "Y must have shape [M, D]");
    TORCH_CHECK(K.dim() == 2, "K must have shape [N, M]");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must have shape [N, M]");
    TORCH_CHECK(X.size(1) == Y.size(1), "X and Y must have the same D");
    TORCH_CHECK(K.size(0) == X.size(0) && K.size(1) == Y.size(0), "K shape mismatch");
    TORCH_CHECK(grad_out.size(0) == X.size(0) && grad_out.size(1) == Y.size(0),
                "grad_out shape mismatch");

    const int N = static_cast<int>(X.size(0));
    const int D = static_cast<int>(X.size(1));
    const int M = static_cast<int>(Y.size(0));

    auto dX = torch::empty_like(X);
    auto dY = torch::empty_like(Y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (D == D64) {
        const float scale = -2.0f * static_cast<float>(gamma);
        rbf_backward_dx_d64_kernel<<<N, THREADS_D64, 0, stream>>>(
            grad_out.data_ptr<float>(),
            K.data_ptr<float>(),
            X.data_ptr<float>(),
            Y.data_ptr<float>(),
            dX.data_ptr<float>(),
            N,
            M,
            scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        rbf_backward_dy_d64_kernel<<<M, THREADS_D64, 0, stream>>>(
            grad_out.data_ptr<float>(),
            K.data_ptr<float>(),
            X.data_ptr<float>(),
            Y.data_ptr<float>(),
            dY.data_ptr<float>(),
            N,
            M,
            scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        const int threads = 256;
        const int dx_blocks = (N * D + threads - 1) / threads;
        const int dy_blocks = (M * D + threads - 1) / threads;
        rbf_backward_dx_kernel<<<dx_blocks, threads, 0, stream>>>(
            grad_out.data_ptr<float>(),
            K.data_ptr<float>(),
            X.data_ptr<float>(),
            Y.data_ptr<float>(),
            dX.data_ptr<float>(),
            N,
            M,
            D,
            static_cast<float>(gamma));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        rbf_backward_dy_kernel<<<dy_blocks, threads, 0, stream>>>(
            grad_out.data_ptr<float>(),
            K.data_ptr<float>(),
            X.data_ptr<float>(),
            Y.data_ptr<float>(),
            dY.data_ptr<float>(),
            N,
            M,
            D,
            static_cast<float>(gamma));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {dX, dY};
}
