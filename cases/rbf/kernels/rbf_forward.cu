#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int TILE_N = 32;
constexpr int TILE_M = 32;
constexpr int TILE_D = 32;
constexpr int THREAD_X = 16;
constexpr int THREAD_Y = 16;
constexpr int OUT_N = TILE_N / THREAD_Y;
constexpr int OUT_M = TILE_M / THREAD_X;

__global__ void rbf_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int n,
    int m,
    int d,
    float gamma) {
    __shared__ float xs[TILE_N][TILE_D];
    __shared__ float ys[TILE_M][TILE_D];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * THREAD_X + tx;
    int row_base = blockIdx.y * TILE_N;
    int col_base = blockIdx.x * TILE_M;

    float dist[OUT_N][OUT_M];
#pragma unroll
    for (int rn = 0; rn < OUT_N; ++rn) {
#pragma unroll
        for (int rm = 0; rm < OUT_M; ++rm) {
            dist[rn][rm] = 0.0f;
        }
    }

    for (int d_base = 0; d_base < d; d_base += TILE_D) {
        for (int idx = tid; idx < TILE_N * TILE_D; idx += THREAD_X * THREAD_Y) {
            int r = idx / TILE_D;
            int c = idx - r * TILE_D;
            int global_r = row_base + r;
            int global_d = d_base + c;
            xs[r][c] = (global_r < n && global_d < d)
                ? x[static_cast<long long>(global_r) * d + global_d]
                : 0.0f;
        }
        for (int idx = tid; idx < TILE_M * TILE_D; idx += THREAD_X * THREAD_Y) {
            int r = idx / TILE_D;
            int c = idx - r * TILE_D;
            int global_r = col_base + r;
            int global_d = d_base + c;
            ys[r][c] = (global_r < m && global_d < d)
                ? y[static_cast<long long>(global_r) * d + global_d]
                : 0.0f;
        }
        __syncthreads();

        int d_limit = min(TILE_D, d - d_base);
#pragma unroll
        for (int rn = 0; rn < OUT_N; ++rn) {
            int x_row = ty + rn * THREAD_Y;
#pragma unroll
            for (int rm = 0; rm < OUT_M; ++rm) {
                int y_row = tx + rm * THREAD_X;
                float acc = 0.0f;
                int k = 0;
                for (; k + 3 < d_limit; k += 4) {
                    float dx0 = xs[x_row][k] - ys[y_row][k];
                    float dx1 = xs[x_row][k + 1] - ys[y_row][k + 1];
                    float dx2 = xs[x_row][k + 2] - ys[y_row][k + 2];
                    float dx3 = xs[x_row][k + 3] - ys[y_row][k + 3];
                    acc += dx0 * dx0 + dx1 * dx1 + dx2 * dx2 + dx3 * dx3;
                }
                for (; k < d_limit; ++k) {
                    float diff = xs[x_row][k] - ys[y_row][k];
                    acc += diff * diff;
                }
                dist[rn][rm] += acc;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int rn = 0; rn < OUT_N; ++rn) {
        int row = row_base + ty + rn * THREAD_Y;
#pragma unroll
        for (int rm = 0; rm < OUT_M; ++rm) {
            int col = col_base + tx + rm * THREAD_X;
            if (row < n && col < m) {
                out[static_cast<long long>(row) * m + col] = expf(-gamma * dist[rn][rm]);
            }
        }
    }
}

}  // namespace

torch::Tensor rbf_forward(torch::Tensor x, torch::Tensor y, double gamma) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "RBF inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && y.scalar_type() == torch::kFloat32,
                "RBF only supports float32");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "RBF inputs must be 2D");
    TORCH_CHECK(x.size(1) == y.size(1), "RBF feature dimensions must match");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "RBF inputs must be contiguous");

    int n = static_cast<int>(x.size(0));
    int d = static_cast<int>(x.size(1));
    int m = static_cast<int>(y.size(0));
    auto out = torch::empty({n, m}, x.options());

    dim3 block(THREAD_X, THREAD_Y);
    dim3 grid((m + TILE_M - 1) / TILE_M, (n + TILE_N - 1) / TILE_N);
    rbf_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        n,
        m,
        d,
        static_cast<float>(gamma));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
