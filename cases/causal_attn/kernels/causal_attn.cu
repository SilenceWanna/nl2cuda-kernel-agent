#include <torch/extension.h>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxT = 2048;

#define CUBLAS_CHECK(expr)                                                        \
    do {                                                                          \
        cublasStatus_t status_ = (expr);                                          \
        TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS,                             \
                    #expr, " failed with cuBLAS status ",                         \
                    static_cast<int>(status_));                                   \
    } while (0)

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float block_sum(float value) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_sum(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ float block_max(float value) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
    if (warp == 0) {
        value = warp_max(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__global__ void causal_softmax_kernel(float* __restrict__ scores, int rows, int T) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int i = row % T;
    float* row_ptr = scores + static_cast<int64_t>(row) * T;

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        local_max = fmaxf(local_max, row_ptr[j]);
    }
    const float row_max = block_max(local_max);

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        const float p = expf(row_ptr[j] - row_max);
        row_ptr[j] = p;
        local_sum += p;
    }
    const float row_sum = block_sum(local_sum);
    const float inv_sum = 1.0f / row_sum;

    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        row_ptr[j] = (j <= i) ? row_ptr[j] * inv_sum : 0.0f;
    }
}

__global__ void causal_ds_kernel(
        const float* __restrict__ probs,
        float* __restrict__ dp,
        int rows,
        int T) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int i = row % T;
    const int64_t base = static_cast<int64_t>(row) * T;

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        local_sum += probs[base + j] * dp[base + j];
    }
    const float delta = block_sum(local_sum);
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        const int64_t idx = base + j;
        dp[idx] = (j <= i) ? probs[idx] * (dp[idx] - delta) : 0.0f;
    }
}

void validate_qkv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 && k.scalar_type() == torch::kFloat32 &&
                v.scalar_type() == torch::kFloat32, "Q/K/V must be float32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "Q/K/V must be contiguous");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "Q/K/V must have shape [B,T,D]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "Q/K/V shapes must match");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0, "dimensions must be positive");
    TORCH_CHECK(q.size(1) <= kMaxT, "T exceeds kernel maximum");
    TORCH_CHECK(q.size(0) <= INT32_MAX && q.size(1) <= INT32_MAX && q.size(2) <= INT32_MAX,
                "Q/K/V dimensions are too large");
}

void validate_grad(const torch::Tensor& grad_out, const torch::Tensor& q) {
    TORCH_CHECK(grad_out.is_cuda() && grad_out.scalar_type() == torch::kFloat32 &&
                grad_out.is_contiguous() && grad_out.sizes() == q.sizes(),
                "grad_out must be contiguous CUDA float32 with shape [B,T,D]");
}

void set_cublas(cublasHandle_t handle, cudaStream_t stream, cublasMath_t* previous_math) {
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasGetMathMode(handle, previous_math));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}

void restore_cublas(cublasHandle_t handle, cublasMath_t previous_math) {
    CUBLAS_CHECK(cublasSetMathMode(handle, previous_math));
}

void batched_gemm(
        cublasHandle_t handle,
        cublasOperation_t op_a,
        cublasOperation_t op_b,
        int m,
        int n,
        int k,
        const float alpha,
        const float* A,
        int lda,
        int64_t stride_a,
        const float* B,
        int ldb,
        int64_t stride_b,
        const float beta,
        float* C,
        int ldc,
        int64_t stride_c,
        int batch_count) {
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, op_a, op_b, m, n, k,
        &alpha,
        A, lda, stride_a,
        B, ldb, stride_b,
        &beta,
        C, ldc, stride_c,
        batch_count));
}

}  // namespace

std::vector<torch::Tensor> causal_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    validate_qkv(q, k, v);
    const int B = static_cast<int>(q.size(0));
    const int T = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    auto probs = torch::empty({B, T, T}, q.options());
    auto out = torch::empty_like(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasMath_t previous_math;
    set_cublas(handle, stream, &previous_math);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const float one = 1.0f;
    const float zero = 0.0f;
    const int64_t qkv_stride = static_cast<int64_t>(T) * D;
    const int64_t prob_stride = static_cast<int64_t>(T) * T;

    batched_gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 T, T, D, scale,
                 k.data_ptr<float>(), D, qkv_stride,
                 q.data_ptr<float>(), D, qkv_stride,
                 zero,
                 probs.data_ptr<float>(), T, prob_stride,
                 B);
    causal_softmax_kernel<<<B * T, kThreads, 0, stream>>>(probs.data_ptr<float>(), B * T, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    batched_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 D, T, T, one,
                 v.data_ptr<float>(), D, qkv_stride,
                 probs.data_ptr<float>(), T, prob_stride,
                 zero,
                 out.data_ptr<float>(), D, qkv_stride,
                 B);
    restore_cublas(handle, previous_math);
    return {out, probs};
}

std::vector<torch::Tensor> causal_attn_backward(
        torch::Tensor grad_out,
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor probs) {
    validate_qkv(q, k, v);
    validate_grad(grad_out, q);
    TORCH_CHECK(probs.is_cuda() && probs.scalar_type() == torch::kFloat32 && probs.is_contiguous() &&
                probs.dim() == 3 && probs.size(0) == q.size(0) && probs.size(1) == q.size(1) &&
                probs.size(2) == q.size(1), "probs must be contiguous [B,T,T]");
    const int B = static_cast<int>(q.size(0));
    const int T = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    auto grad_q = torch::empty_like(q);
    auto grad_k = torch::empty_like(k);
    auto grad_v = torch::empty_like(v);
    auto dp = torch::empty_like(probs);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasMath_t previous_math;
    set_cublas(handle, stream, &previous_math);

    const float one = 1.0f;
    const float zero = 0.0f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int64_t qkv_stride = static_cast<int64_t>(T) * D;
    const int64_t prob_stride = static_cast<int64_t>(T) * T;

    batched_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                 D, T, T, one,
                 grad_out.data_ptr<float>(), D, qkv_stride,
                 probs.data_ptr<float>(), T, prob_stride,
                 zero,
                 grad_v.data_ptr<float>(), D, qkv_stride,
                 B);

    batched_gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 T, T, D, one,
                 v.data_ptr<float>(), D, qkv_stride,
                 grad_out.data_ptr<float>(), D, qkv_stride,
                 zero,
                 dp.data_ptr<float>(), T, prob_stride,
                 B);

    causal_ds_kernel<<<B * T, kThreads, 0, stream>>>(
        probs.data_ptr<float>(), dp.data_ptr<float>(), B * T, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    batched_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 D, T, T, scale,
                 k.data_ptr<float>(), D, qkv_stride,
                 dp.data_ptr<float>(), T, prob_stride,
                 zero,
                 grad_q.data_ptr<float>(), D, qkv_stride,
                 B);

    batched_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                 D, T, T, scale,
                 q.data_ptr<float>(), D, qkv_stride,
                 dp.data_ptr<float>(), T, prob_stride,
                 zero,
                 grad_k.data_ptr<float>(), D, qkv_stride,
                 B);

    restore_cublas(handle, previous_math);
    return {grad_q, grad_k, grad_v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("causal_attn_forward", &causal_attn_forward, "Causal attention forward (CUDA)");
    module.def("causal_attn_backward", &causal_attn_backward, "Causal attention backward (CUDA)");
}
