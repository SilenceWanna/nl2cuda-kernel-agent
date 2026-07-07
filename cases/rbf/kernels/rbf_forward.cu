// RBF 高斯核矩阵——前向 kernel（阶段3优化版：shared-memory tiling）。
//   X:[N,D], Y:[M,D]；Dist[i,j]=sum_d (X[i,d]-Y[j,d])^2；K[i,j]=exp(-gamma*Dist[i,j])
//
// 优化：GEMM 式分块。一个 block 计算 K 的 TILE×TILE 子块；沿 D 维分块，
// block 内协作把 X 的 TILE×TILE_D 与 Y 的 TILE×TILE_D 子块载入 shared memory 复用，
// 全局内存读取量相比朴素版（每个 K 元素各读整行 X/Y）减少约 TILE 倍。
// 保持 fp32 全精度（不降精度、不 fast-math）。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void rbf_forward_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        float* __restrict__ K,         // [N, M]
        int N, int M, int D, float gamma) {
    __shared__ float Xs[TILE][TILE];   // [row_in_tile][d_in_tile]
    __shared__ float Ys[TILE][TILE];   // [col_in_tile][d_in_tile]

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;  // K 的行 i (对应 X 行)
    int col = blockIdx.x * TILE + tx;  // K 的列 j (对应 Y 行)

    float dist = 0.0f;
    for (int d0 = 0; d0 < D; d0 += TILE) {
        // 协作载入：Xs[ty][tx] = X[(blockIdx.y*TILE+ty), d0+tx]
        int xd = d0 + tx;
        Xs[ty][tx] = (row < N && xd < D) ? X[(size_t)row * D + xd] : 0.0f;
        // Ys[ty][tx] = Y[(blockIdx.x*TILE+ty), d0+tx]
        int yrow = blockIdx.x * TILE + ty;
        int yd = d0 + tx;
        Ys[ty][tx] = (yrow < M && yd < D) ? Y[(size_t)yrow * D + yd] : 0.0f;
        __syncthreads();

        // 用 shared 里的子块累加平方差：线程(ty,tx) 用 Xs[ty][.] 与 Ys[tx][.]
        int dlim = min(TILE, D - d0);
        for (int dd = 0; dd < dlim; ++dd) {
            float diff = Xs[ty][dd] - Ys[tx][dd];
            dist += diff * diff;
        }
        __syncthreads();
    }

    if (row < N && col < M) {
        K[(size_t)row * M + col] = expf(-gamma * dist);
    }
}

torch::Tensor rbf_forward(torch::Tensor X, torch::Tensor Y, double gamma) {
    TORCH_CHECK(X.is_cuda() && Y.is_cuda(), "X, Y must be CUDA tensors");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && Y.dtype() == torch::kFloat32,
                "X, Y must be float32");
    TORCH_CHECK(X.dim() == 2 && Y.dim() == 2, "X, Y must be 2D");
    TORCH_CHECK(X.size(1) == Y.size(1), "X, Y must share feature dim D");

    X = X.contiguous();
    Y = Y.contiguous();
    int N = X.size(0);
    int M = Y.size(0);
    int D = X.size(1);

    auto K = torch::empty({N, M}, X.options());

    dim3 threads(TILE, TILE);
    dim3 blocks((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    rbf_forward_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), K.data_ptr<float>(),
        N, M, D, (float)gamma);

    return K;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbf_forward", &rbf_forward, "RBF kernel matrix forward (CUDA, shared-mem tiled)");
}
