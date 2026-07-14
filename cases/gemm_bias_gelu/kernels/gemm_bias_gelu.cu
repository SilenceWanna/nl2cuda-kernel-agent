#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int TILE = 16;
constexpr int THREADS = 256;

constexpr float GELU_C = 0.7978845608028654f;
constexpr float GELU_A = 0.044715f;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT32(x)

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

__global__ void gemm_bias_gelu_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    float* __restrict__ z,
    int m_size,
    int k_size,
    int n_size
) {
    __shared__ float sx[TILE][TILE];
    __shared__ float sw[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < k_size; t += TILE) {
        int x_col = t + threadIdx.x;
        int w_row = t + threadIdx.y;

        sx[threadIdx.y][threadIdx.x] =
            (row < m_size && x_col < k_size) ? x[row * k_size + x_col] : 0.0f;

        sw[threadIdx.y][threadIdx.x] =
            (w_row < k_size && col < n_size) ? w[w_row * n_size + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += sx[threadIdx.y][i] * sw[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m_size && col < n_size) {
        float v = acc + b[col];
        z[row * n_size + col] = v;
        y[row * n_size + col] = gelu_tanh(v);
    }
}

__global__ void gelu_backward_pointwise_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ z,
    float* __restrict__ grad_z,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grad_z[idx] = grad_y[idx] * gelu_tanh_grad(z[idx]);
    }
}

__global__ void grad_x_kernel(
    const float* __restrict__ grad_z,
    const float* __restrict__ w,
    float* __restrict__ grad_x,
    int m_size,
    int k_size,
    int n_size
) {
    __shared__ float sgz[TILE][TILE];
    __shared__ float swt[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < n_size; t += TILE) {
        int n_col = t + threadIdx.x;
        int n_row = t + threadIdx.y;

        sgz[threadIdx.y][threadIdx.x] =
            (row < m_size && n_col < n_size) ? grad_z[row * n_size + n_col] : 0.0f;

        swt[threadIdx.y][threadIdx.x] =
            (col < k_size && n_row < n_size) ? w[col * n_size + n_row] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += sgz[threadIdx.y][i] * swt[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m_size && col < k_size) {
        grad_x[row * k_size + col] = acc;
    }
}

__global__ void grad_w_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_z,
    float* __restrict__ grad_w,
    int m_size,
    int k_size,
    int n_size
) {
    __shared__ float sxt[TILE][TILE];
    __shared__ float sgz[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < m_size; t += TILE) {
        int m_col = t + threadIdx.x;
        int m_row = t + threadIdx.y;

        sxt[threadIdx.y][threadIdx.x] =
            (row < k_size && m_col < m_size) ? x[m_col * k_size + row] : 0.0f;

        sgz[threadIdx.y][threadIdx.x] =
            (m_row < m_size && col < n_size) ? grad_z[m_row * n_size + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += sxt[threadIdx.y][i] * sgz[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < k_size && col < n_size) {
        grad_w[row * n_size + col] = acc;
    }
}

__global__ void grad_b_kernel(
    const float* __restrict__ grad_z,
    float* __restrict__ grad_b,
    int m_size,
    int n_size
) {
    int col = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float partial[THREADS];

    float acc = 0.0f;
    for (int row = tid; row < m_size; row += blockDim.x) {
        acc += grad_z[row * n_size + col];
    }

    partial[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_b[col] = partial[0];
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

    auto y = torch::empty({m_size, n_size}, x.options());
    auto z = torch::empty({m_size, n_size}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((n_size + TILE - 1) / TILE, (m_size + TILE - 1) / TILE);

    gemm_bias_gelu_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        z.data_ptr<float>(),
        m_size,
        k_size,
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
    int total = m_size * n_size;

    auto grad_z = torch::empty({m_size, n_size}, x.options());
    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(w);
    auto grad_b = torch::empty({n_size}, x.options());

    int pointwise_blocks = (total + THREADS - 1) / THREADS;
    gelu_backward_pointwise_kernel<<<pointwise_blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_y.data_ptr<float>(),
        z.data_ptr<float>(),
        grad_z.data_ptr<float>(),
        total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 block(TILE, TILE);

    dim3 grid_x((k_size + TILE - 1) / TILE, (m_size + TILE - 1) / TILE);
    grad_x_kernel<<<grid_x, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_z.data_ptr<float>(),
        w.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        m_size,
        k_size,
        n_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_w((n_size + TILE - 1) / TILE, (k_size + TILE - 1) / TILE);
    grad_w_kernel<<<grid_w, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        grad_z.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        m_size,
        k_size,
        n_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    grad_b_kernel<<<n_size, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_z.data_ptr<float>(),
        grad_b.data_ptr<float>(),
        m_size,
        n_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_w, grad_b};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_bias_gelu_forward", &gemm_bias_gelu_forward, "GEMM + bias + GELU forward");
    m.def("gemm_bias_gelu_backward", &gemm_bias_gelu_backward, "GEMM + bias + GELU backward");
}
