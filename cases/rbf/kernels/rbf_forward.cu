// RBF 前向 kernel（阶段3：tiling + coarsening + float4，精确 expf）。
//   X:[N,D], Y:[M,D]；Dist[i,j]=sum_d (X[i,d]-Y[j,d])^2；K[i,j]=exp(-gamma*Dist[i,j])
//
// 优化历程：朴素4.8ms → tiling(1024线程)4.5 → coarsening(256线程/block)2.34 → +float4 2.27ms。
// 主要收益来自提高 occupancy（256线程/block + 每线程2×2输出微块）。
// 经测量前向瓶颈非 expf（__expf 不提速），保持精确 expf（防作弊最干净）。fp32 全精度。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BN 32
#define BM 32
#define TD 32
#define TX 16
#define TY 16
#define RN (BN / TY)
#define RM (BM / TX)

__global__ void rbf_forward_kernel(
        const float* __restrict__ X,
        const float* __restrict__ Y,
        float* __restrict__ K,
        int N, int M, int D, float gamma) {
    __shared__ float Xs[BN][TD];
    __shared__ float Ys[BM][TD];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row0 = blockIdx.y * BN;
    int col0 = blockIdx.x * BM;

    float dist[RN][RM];
    #pragma unroll
    for (int a = 0; a < RN; ++a)
        for (int b = 0; b < RM; ++b) dist[a][b] = 0.0f;

    for (int d0 = 0; d0 < D; d0 += TD) {
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
            int xr = ty + a * TY;
            const float4* xs4 = reinterpret_cast<const float4*>(&Xs[xr][0]);
            #pragma unroll
            for (int b = 0; b < RM; ++b) {
                int yr = tx + b * TX;
                const float4* ys4 = reinterpret_cast<const float4*>(&Ys[yr][0]);
                float acc = 0.0f;
                if (dlim == TD) {
                    #pragma unroll
                    for (int q = 0; q < TD / 4; ++q) {
                        float4 xv = xs4[q], yv = ys4[q];
                        float d0f = xv.x - yv.x, d1f = xv.y - yv.y;
                        float d2f = xv.z - yv.z, d3f = xv.w - yv.w;
                        acc += d0f*d0f + d1f*d1f + d2f*d2f + d3f*d3f;
                    }
                } else {
                    for (int dd = 0; dd < dlim; ++dd) {
                        float diff = Xs[xr][dd] - Ys[yr][dd];
                        acc += diff * diff;
                    }
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
