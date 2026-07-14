#include <torch/extension.h>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int FORWARD_THREADS = 256;
constexpr int DZ_THREADS = 256;
constexpr int DZ_ROWS_PER_BLOCK = 64;
constexpr float GELU_COEFF = 0.7978845608028654f;
constexpr float GELU_CUBIC = 0.044715f;

#define CUBLAS_CHECK(expr)                                                        \
    do {                                                                          \
        cublasStatus_t status_ = (expr);                                           \
        TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS,                              \
                    #expr, " failed with cuBLAS status ",                         \
                    static_cast<int>(status_));                                    \
    } while (0)

__device__ __forceinline__ float gelu_tanh(float z) {
    const float z2 = z * z;
    const float u = GELU_COEFF * (z + GELU_CUBIC * z * z2);
    return 0.5f * z * (1.0f + tanhf(u));
}

__global__ void bias_gelu_forward_float4_kernel(
    const float* __restrict__ gemm_output,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int n_vectors) {
    const int col_vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_vector >= n_vectors) {
        return;
    }

    const int64_t index = static_cast<int64_t>(blockIdx.y) * n_vectors + col_vector;
    float4 z = reinterpret_cast<const float4*>(gemm_output)[index];
    const float4 b = reinterpret_cast<const float4*>(bias)[col_vector];
    z.x += b.x;
    z.y += b.y;
    z.z += b.z;
    z.w += b.w;
    reinterpret_cast<float4*>(out)[index] = {
        gelu_tanh(z.x), gelu_tanh(z.y), gelu_tanh(z.z), gelu_tanh(z.w)};
}

__global__ void bias_gelu_forward_scalar_kernel(
    const float* __restrict__ gemm_output,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t elements,
    int n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float z = gemm_output[index] + bias[index % n];
        out[index] = gelu_tanh(z);
    }
}

__global__ void gelu_backward_bias_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ gemm_output,
    const float* __restrict__ bias,
    float* __restrict__ grad_z,
    float* __restrict__ grad_bias,
    int m,
    int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }

    const int row_begin = blockIdx.y * DZ_ROWS_PER_BLOCK;
    const int row_end = min(row_begin + DZ_ROWS_PER_BLOCK, m);
    float bias_sum = 0.0f;
    for (int row = row_begin; row < row_end; ++row) {
        const int64_t offset = static_cast<int64_t>(row) * n + col;
        const float z = gemm_output[offset] + bias[col];
        const float z2 = z * z;
        const float u = GELU_COEFF * (z + GELU_CUBIC * z * z2);
        const float t = tanhf(u);
        const float derivative = 0.5f * (1.0f + t)
            + 0.5f * z * (1.0f - t * t) * GELU_COEFF
                * (1.0f + 3.0f * GELU_CUBIC * z2);
        const float dz = grad_out[offset] * derivative;
        grad_z[offset] = dz;
        bias_sum += dz;
    }
    atomicAdd(grad_bias + col, bias_sum);
}

void check_forward_inputs(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& b) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(),
                "GEMM+bias+GELU inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                w.scalar_type() == torch::kFloat32 &&
                b.scalar_type() == torch::kFloat32,
                "GEMM+bias+GELU only supports float32");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && b.dim() == 1,
                "expected X[M,K], W[K,N], b[N]");
    TORCH_CHECK(x.size(1) == w.size(0) && w.size(1) == b.size(0),
                "GEMM+bias+GELU shape mismatch");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && b.is_contiguous(),
                "GEMM+bias+GELU inputs must be contiguous");
    TORCH_CHECK(x.device() == w.device() && x.device() == b.device(),
                "GEMM+bias+GELU inputs must be on one device");
}

}  // namespace

std::vector<torch::Tensor> gemm_bias_gelu_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b) {
    check_forward_inputs(x, w, b);
    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t n = w.size(1);
    auto out = torch::empty({m, n}, x.options());
    auto gemm_output = torch::empty({m, n}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    cublasMath_t previous_math;
    CUBLAS_CHECK(cublasGetMathMode(handle, &previous_math));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major Z = X @ W, expressed as column-major Z^T = W^T @ X^T.
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
        &alpha,
        w.data_ptr<float>(), static_cast<int>(n),
        x.data_ptr<float>(), static_cast<int>(k),
        &beta,
        gemm_output.data_ptr<float>(), static_cast<int>(n)));
    CUBLAS_CHECK(cublasSetMathMode(handle, previous_math));

    const int64_t elements = m * n;
    if ((n & 3) == 0) {
        const int n_vectors = static_cast<int>(n / 4);
        dim3 forward_grid(
            (n_vectors + FORWARD_THREADS - 1) / FORWARD_THREADS,
            static_cast<unsigned int>(m));
        bias_gelu_forward_float4_kernel<<<
            forward_grid,
            FORWARD_THREADS,
            0,
            stream>>>(
                gemm_output.data_ptr<float>(),
                b.data_ptr<float>(),
                out.data_ptr<float>(),
                n_vectors);
    } else {
        bias_gelu_forward_scalar_kernel<<<
            (elements + FORWARD_THREADS - 1) / FORWARD_THREADS,
            FORWARD_THREADS,
            0,
            stream>>>(
                gemm_output.data_ptr<float>(),
                b.data_ptr<float>(),
                out.data_ptr<float>(),
                elements,
                static_cast<int>(n));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, gemm_output};
}

std::vector<torch::Tensor> gemm_bias_gelu_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor gemm_output) {
    TORCH_CHECK(grad_out.is_cuda() && b.is_cuda() && gemm_output.is_cuda(),
                "GEMM+bias+GELU backward tensors must be CUDA tensors");
    TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 &&
                b.scalar_type() == torch::kFloat32 &&
                gemm_output.scalar_type() == torch::kFloat32,
                "GEMM+bias+GELU backward only supports float32");
    TORCH_CHECK(grad_out.is_contiguous() && x.is_contiguous() &&
                w.is_contiguous() && b.is_contiguous() && gemm_output.is_contiguous(),
                "GEMM+bias+GELU backward tensors must be contiguous");
    TORCH_CHECK(grad_out.sizes() == gemm_output.sizes(),
                "saved GEMM output shape mismatch");
    TORCH_CHECK(grad_out.size(0) == x.size(0) &&
                grad_out.size(1) == w.size(1) && x.size(1) == w.size(0),
                "GEMM+bias+GELU backward shape mismatch");

    const int m = static_cast<int>(x.size(0));
    const int k = static_cast<int>(x.size(1));
    const int n = static_cast<int>(w.size(1));
    auto grad_z = torch::empty_like(grad_out);
    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(w);
    auto grad_b = torch::empty({n}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaMemsetAsync(
        grad_b.data_ptr<float>(), 0, static_cast<size_t>(n) * sizeof(float), stream));
    dim3 block(DZ_THREADS);
    dim3 grid(
        (n + DZ_THREADS - 1) / DZ_THREADS,
        (m + DZ_ROWS_PER_BLOCK - 1) / DZ_ROWS_PER_BLOCK);
    gelu_backward_bias_kernel<<<grid, block, 0, stream>>>(
        grad_out.data_ptr<float>(),
        gemm_output.data_ptr<float>(),
        b.data_ptr<float>(),
        grad_z.data_ptr<float>(),
        grad_b.data_ptr<float>(),
        m,
        n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    cublasMath_t previous_math;
    CUBLAS_CHECK(cublasGetMathMode(handle, &previous_math));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major dX = dZ @ W^T, expressed as column-major dX^T = W @ dZ^T.
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        k, m, n,
        &alpha,
        w.data_ptr<float>(), n,
        grad_z.data_ptr<float>(), n,
        &beta,
        grad_x.data_ptr<float>(), k));

    // Row-major dW = X^T @ dZ, expressed as column-major dW^T = dZ^T @ X.
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        n, k, m,
        &alpha,
        grad_z.data_ptr<float>(), n,
        x.data_ptr<float>(), k,
        &beta,
        grad_w.data_ptr<float>(), n));
    CUBLAS_CHECK(cublasSetMathMode(handle, previous_math));

    return {grad_x, grad_w, grad_b};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_bias_gelu_forward", &gemm_bias_gelu_forward,
          "Fused GEMM+bias+tanh-GELU forward (CUDA)");
    m.def("gemm_bias_gelu_backward", &gemm_bias_gelu_backward,
          "Fused GEMM+bias+tanh-GELU backward (CUDA)");
}
