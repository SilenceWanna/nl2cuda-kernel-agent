#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int THREADS = 256;
constexpr int COL_TILE = 32;
constexpr int ROW_THREADS = 8;
constexpr int ROW_TILE = 256;

constexpr float GELU_C = 0.7978845608028654f;
constexpr float GELU_A = 0.044715f;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x)      \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x);    \
    CHECK_FLOAT32(x)

#define CHECK_CUBLAS(status) TORCH_CHECK((status) == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed")

__device__ __forceinline__ float gelu_tanh(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float u = GELU_C * (x + GELU_A * x3);
    float t = tanhf(u);
    return 0.5f * x * (1.0f + t);
}

__device__ __forceinline__ float gelu_tanh_grad(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float u = GELU_C * (x + GELU_A * x3);
    float t = tanhf(u);
    float sech2 = 1.0f - t * t;
    float du = GELU_C * (1.0f + 3.0f * GELU_A * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * du;
}

__global__ void bias_gelu_from_matmul_kernel(
    const float* __restrict__ b,
    float* __restrict__ z,
    float* __restrict__ y,
    int total,
    int n_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int col = idx % n_size;
        float v = z[idx] + b[col];
        z[idx] = v;
        y[idx] = gelu_tanh(v);
    }
}

__global__ void gelu_backward_and_bias_grad_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ z,
    float* __restrict__ grad_z,
    float* __restrict__ grad_b,
    int m_size,
    int n_size
) {
    __shared__ float partial[ROW_THREADS][COL_TILE];

    int col = blockIdx.x * COL_TILE + threadIdx.x;
    int row_start = blockIdx.y * ROW_TILE;
    float acc = 0.0f;

    if (col < n_size) {
        for (int r = threadIdx.y; r < ROW_TILE; r += ROW_THREADS) {
            int row = row_start + r;
            if (row < m_size) {
                int idx = row * n_size + col;
                float gz = grad_y[idx] * gelu_tanh_grad(z[idx]);
                grad_z[idx] = gz;
                acc += gz;
            }
        }
    }

    partial[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.y == 0 && col < n_size) {
        float sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < ROW_THREADS; ++i) {
            sum += partial[i][threadIdx.x];
        }

        atomicAdd(grad_b + col, sum);
    }
}

void check_forward_inputs(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& b) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(b);

    TORCH_CHECK(x.dim() == 2, "X must be 2D");
    TORCH_CHECK(w.dim() == 2, "W must be 2D");
    TORCH_CHECK(b.dim() == 1, "b must be 1D");

    TORCH_CHECK(x.size(1) == w.size(0), "X.shape[1] must equal W.shape[0]");
    TORCH_CHECK(w.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
}

void check_backward_inputs(
    const torch::Tensor& grad_y,
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& z
) {
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(z);

    TORCH_CHECK(grad_y.dim() == 2, "grad_y must be 2D");
    TORCH_CHECK(x.dim() == 2, "X must be 2D");
    TORCH_CHECK(w.dim() == 2, "W must be 2D");
    TORCH_CHECK(z.dim() == 2, "Z must be 2D");

    TORCH_CHECK(x.size(1) == w.size(0), "X.shape[1] must equal W.shape[0]");
    TORCH_CHECK(grad_y.size(0) == x.size(0), "grad_y.shape[0] must equal X.shape[0]");
    TORCH_CHECK(grad_y.size(1) == w.size(1), "grad_y.shape[1] must equal W.shape[1]");
    TORCH_CHECK(z.size(0) == grad_y.size(0), "Z.shape[0] must equal grad_y.shape[0]");
    TORCH_CHECK(z.size(1) == grad_y.size(1), "Z.shape[1] must equal grad_y.shape[1]");
}

cublasHandle_t current_cublas_handle() {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    CHECK_CUBLAS(cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    return handle;
}

void row_major_gemm_nn(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b,
        n,
        a,
        k,
        &beta,
        c,
        n
    ));
}

void row_major_gemm_nt(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b,
        k,
        a,
        k,
        &beta,
        c,
        n
    ));
}

void row_major_gemm_tn(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        n,
        m,
        k,
        &alpha,
        b,
        n,
        a,
        m,
        &beta,
        c,
        n
    ));
}

}  // namespace

std::vector<torch::Tensor> gemm_bias_gelu_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b
) {
    check_forward_inputs(x, w, b);

    int m_size = static_cast<int>(x.size(0));
    int k_size = static_cast<int>(x.size(1));
    int n_size = static_cast<int>(w.size(1));
    int total = m_size * n_size;

    auto y = torch::empty({m_size, n_size}, x.options());
    auto z = torch::empty({m_size, n_size}, x.options());

    cublasHandle_t handle = current_cublas_handle();

    row_major_gemm_nn(
        handle,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        z.data_ptr<float>(),
        m_size,
        n_size,
        k_size
    );

    int blocks = (total + THREADS - 1) / THREADS;
    bias_gelu_from_matmul_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        b.data_ptr<float>(),
        z.data_ptr<float>(),
        y.data_ptr<float>(),
        total,
        n_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {y, z};
}

std::vector<torch::Tensor> gemm_bias_gelu_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor z
) {
    check_backward_inputs(grad_y, x, w, z);

    int m_size = static_cast<int>(x.size(0));
    int k_size = static_cast<int>(x.size(1));
    int n_size = static_cast<int>(w.size(1));

    auto grad_z = torch::empty({m_size, n_size}, x.options());
    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(w);
    auto grad_b = torch::empty({n_size}, x.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(grad_b.data_ptr<float>(), 0, n_size * sizeof(float), stream));

    dim3 db_block(COL_TILE, ROW_THREADS);
    dim3 db_grid((n_size + COL_TILE - 1) / COL_TILE, (m_size + ROW_TILE - 1) / ROW_TILE);
    gelu_backward_and_bias_grad_kernel<<<db_grid, db_block, 0, stream>>>(
        grad_y.data_ptr<float>(),
        z.data_ptr<float>(),
        grad_z.data_ptr<float>(),
        grad_b.data_ptr<float>(),
        m_size,
        n_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = current_cublas_handle();

    row_major_gemm_nt(
        handle,
        grad_z.data_ptr<float>(),
        w.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        m_size,
        k_size,
        n_size
    );

    row_major_gemm_tn(
        handle,
        x.data_ptr<float>(),
        grad_z.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        k_size,
        n_size,
        m_size
    );

    return {grad_x, grad_w, grad_b};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_bias_gelu_forward", &gemm_bias_gelu_forward, "GEMM + bias + GELU forward");
    m.def("gemm_bias_gelu_backward", &gemm_bias_gelu_backward, "GEMM + bias + GELU backward");
}
