// RBF 前向 kernel（阶段3优化版 v2：高 occupancy tiling + thread coarsening）。
//   X:[N,D], Y:[M,D]；Dist[i,j]=sum_d (X[i,d]-Y[j,d])^2；K[i,j]=exp(-gamma*Dist[i,j])
//
// v1(TILE=32, 1024线程/block) 因 occupancy~50% 未提速。v2 改用：
//   - block = 16×16 = 256 线程（T4 每 SM 可驻留多个 block，occupancy 高）
//   - 每线程算 2×2 输出微块（thread coarsening），一个 block 覆盖 32×32 输出
//   - 沿 D 分块协作载入 X/Y 的 32×TILE_D 子块到 shared memory 复用
// 保持 fp32 全精度（不降精度、不 fast-math）。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BN 32          // block 覆盖的输出行数 (X 方向)
#define BM 32          // block 覆盖的输出列数 (Y 方向)
#define TD 32          // D 方向分块大小
#define TX 16          // blockDim.x
#define TY 16          // blockDim.y
#define RN (BN / TY)   // 每线程负责的行数 = 2
#define RM (BM / TX)   // 每线程负责的列数 = 2

__global__ void rbf_forward_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        float* __restrict__ K,         // [N, M]
        int N, int M, int D, float gamma) {
    __shared__ float Xs[BN][TD];   // 32 x 32
    __shared__ float Ys[BM][TD];   // 32 x 32

    int ty = threadIdx.y, tx = threadIdx.x;
    int row0 = blockIdx.y * BN;    // 本 block 输出行起点
    int col0 = blockIdx.x * BM;    // 本 block 输出列起点

    float dist[RN][RM];
    #pragma unroll
    for (int a = 0; a < RN; ++a)
        for (int b = 0; b < RM; ++b) dist[a][b] = 0.0f;

    for (int d0 = 0; d0 < D; d0 += TD) {
        // 协作载入 Xs[BN][TD]：256 线程搬 32*32=1024 元素，每线程 4 个
        #pragma unroll
        for (int idx = ty * TX + tx; idx < BN * TD; idx += TX * TY) {
            int r = idx / TD, c = idx % TD;
            int gr = row0 + r, gd = d0 + c;
            Xs[r][c] = (gr < N && gd < D) ? X[(size_t)gr * D + gd] : 0.0f;
        }
        #pragma unroll
        for (int idx = ty * TX + tx; idx < BM * TD; idx += TX * TY) {
            int r = idx / TD, c = idx % TD;
            int gr = col0 + r, gd = d0 + c;
            Ys[r][c] = (gr < M && gd < D) ? Y[(size_t)gr * D + gd] : 0.0f;
        }
        __syncthreads();

        int dlim = min(TD, D - d0);
        #pragma unroll
        for (int a = 0; a < RN; ++a) {
            int xr = ty + a * TY;           // Xs 行 (0..31)
            #pragma unroll
            for (int b = 0; b < RM; ++b) {
                int yr = tx + b * TX;       // Ys 行 (0..31)
                float acc = 0.0f;
                for (int dd = 0; dd < dlim; ++dd) {
                    float diff = Xs[xr][dd] - Ys[yr][dd];
                    acc += diff * diff;
                }
                dist[a][b] += acc;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int a = 0; a < RN; ++a) {
        int gr = row0 + ty + a * TY;
        #pragma unroll
        for (int b = 0; b < RM; ++b) {
            int gc = col0 + tx + b * TX;
            if (gr < N && gc < M)
                K[(size_t)gr * M + gc] = expf(-gamma * dist[a][b]);
        }
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

    dim3 threads(TX, TY);
    dim3 blocks((M + BM - 1) / BM, (N + BN - 1) / BN);
    rbf_forward_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), K.data_ptr<float>(),
        N, M, D, (float)gamma);

    return K;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbf_forward", &rbf_forward, "RBF kernel matrix forward (CUDA, tiled+coarsened)");
}
