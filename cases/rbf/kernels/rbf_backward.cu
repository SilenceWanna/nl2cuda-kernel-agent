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


__global__ void rbf_make_a_kernel(
    const float* __restrict__ grad_k,
    const float* __restrict__ k,
    float* __restrict__ a,
    int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    a[idx] = grad_k[idx] * k[idx];
}


__global__ void rbf_row_sum_kernel(
    const float* __restrict__ a,
    float* __restrict__ row_sum,
    int N,
    int M) {
    extern __shared__ float smem[];

    const int i = blockIdx.x;
    const int tid = threadIdx.x;

    if (i >= N) {
        return;
    }

    float acc = 0.0f;
    const int base = i * M;

    for (int j = tid; j < M; j += blockDim.x) {
        acc += a[base + j];
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_sum[i] = smem[0];
    }
}


__global__ void rbf_col_sum_kernel(
    const float* __restrict__ a,
    float* __restrict__ col_sum,
    int N,
    int M) {
    extern __shared__ float smem[];

    const int j = blockIdx.x;
    const int tid = threadIdx.x;

    if (j >= M) {
        return;
    }

    float acc = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        acc += a[i * M + j];
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        col_sum[j] = smem[0];
    }
}


__global__ void rbf_dx_kernel(
    const float* __restrict__ a,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ row_sum,
    float* __restrict__ grad_x,
    int N,
    int M,
    int D,
    float gamma) {
    extern __shared__ float smem[];

    const int i = blockIdx.x;
    const int d = blockIdx.y;
    const int tid = threadIdx.x;

    if (i >= N || d >= D) {
        return;
    }

    float acc = 0.0f;
    const int a_base = i * M;

    for (int j = tid; j < M; j += blockDim.x) {
        acc += a[a_base + j] * y[j * D + d];
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float xd = x[i * D + d];
        grad_x[i * D + d] = -2.0f * gamma * (xd * row_sum[i] - smem[0]);
    }
}


__global__ void rbf_dy_kernel(
    const float* __restrict__ a,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ col_sum,
    float* __restrict__ grad_y,
    int N,
    int M,
    int D,
    float gamma) {
    extern __shared__ float smem[];

    const int j = blockIdx.x;
    const int d = blockIdx.y;
    const int tid = threadIdx.x;

    if (j >= M || d >= D) {
        return;
    }

    float acc = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        acc += a[i * M + j] * x[i * D + d];
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float yd = y[j * D + d];
        grad_y[j * D + d] = 2.0f * gamma * (smem[0] - yd * col_sum[j]);
    }
}


std::vector<torch::Tensor> rbf_backward(
    torch::Tensor grad_k,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor k,
    double gamma) {
    CHECK_INPUT(grad_k);
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(k);

    TORCH_CHECK(x.dim() == 2, "X must have shape [N, D]");
    TORCH_CHECK(y.dim() == 2, "Y must have shape [M, D]");
    TORCH_CHECK(k.dim() == 2, "K must have shape [N, M]");
    TORCH_CHECK(grad_k.dim() == 2, "grad_k must have shape [N, M]");

    const int64_t N64 = x.size(0);
    const int64_t D64 = x.size(1);
    const int64_t M64 = y.size(0);

    TORCH_CHECK(y.size(1) == D64, "X and Y must have the same D");
    TORCH_CHECK(k.size(0) == N64, "K N mismatch");
    TORCH_CHECK(k.size(1) == M64, "K M mismatch");
    TORCH_CHECK(grad_k.size(0) == N64, "grad_k N mismatch");
    TORCH_CHECK(grad_k.size(1) == M64, "grad_k M mismatch");
    TORCH_CHECK(N64 <= static_cast<int64_t>(2147483647), "N is too large");
    TORCH_CHECK(M64 <= static_cast<int64_t>(2147483647), "M is too large");
    TORCH_CHECK(D64 <= static_cast<int64_t>(2147483647), "D is too large");
    TORCH_CHECK(N64 * M64 <= static_cast<int64_t>(2147483647), "N * M is too large");

    const int N = static_cast<int>(N64);
    const int M = static_cast<int>(M64);
    const int D = static_cast<int>(D64);
    const int total = static_cast<int>(N64 * M64);

    auto grad_x = torch::empty_like(x);
    auto grad_y = torch::empty_like(y);
    auto a = torch::empty_like(k);
    auto row_sum = torch::empty({N64}, x.options());
    auto col_sum = torch::empty({M64}, x.options());

    const int threads = 256;
    const size_t shared_bytes = threads * sizeof(float);

    const int a_blocks = (total + threads - 1) / threads;
    rbf_make_a_kernel<<<a_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_k.data_ptr<float>(),
        k.data_ptr<float>(),
        a.data_ptr<float>(),
        total);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    rbf_row_sum_kernel<<<N, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        N,
        M);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    rbf_col_sum_kernel<<<M, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(),
        col_sum.data_ptr<float>(),
        N,
        M);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 dx_blocks(N, D);
    rbf_dx_kernel<<<dx_blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        N,
        M,
        D,
        static_cast<float>(gamma));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 dy_blocks(M, D);
    rbf_dy_kernel<<<dy_blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        col_sum.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        N,
        M,
        D,
        static_cast<float>(gamma));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_y};
}
