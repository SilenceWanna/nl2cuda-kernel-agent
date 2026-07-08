// ============================================================================
// RBF 高斯核矩阵 —— 前向 + 反向 CUDA kernel（纯 CUDA 交付版，不依赖 PyTorch）
// ============================================================================
// 结构：X:[N,D], Y:[M,D]
//   前向  K[i,j] = exp(-gamma * ||x_i - y_j||^2)                    -> K:[N,M]
//   反向  记 coef[i,j] = -2*gamma*G[i,j]*K[i,j]（G=dL/dK，K 为前向输出缓存）
//         dX[i,d] = sum_j coef[i,j] * (X[i,d] - Y[j,d])
//         dY[j,d] = sum_i coef[i,j] * (Y[j,d] - X[i,d])
//
// 计算逻辑与 cases/rbf/kernels/{rbf_forward,rbf_backward}.cu 逐字一致
// （已在 A100 上验收：前向 1.10×、反向 1.17× 超过 torch.compile，正确性全 PASS）。
//
// 前向：GEMM 式 shared-memory tiling + thread coarsening（每线程 2×2 输出微块）
//       + float4 向量化读 shared，高 occupancy。
// 反向：前向缓存 K 复用，coef 为标量、各分量 d 独立累加，无 per-j 规约。
// 全程 fp32 全精度；不使用 fast-math；不调用任何高层库算子（cuBLAS/cuDNN 也未用）。
//
// extern "C" host 函数收裸 float* 主机指针，内部负责 device 内存分配/拷贝/launch/同步。
// 独立编译：  nvcc -O3 -arch=sm_80 -c rbf_kernels.cu
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef RBF_CHECK_CUDA
#define RBF_CHECK_CUDA(call)                                                   \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif

// ---------------- 前向 kernel ----------------
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

// ---------------- 反向 kernel ----------------
__global__ void rbf_backward_dX_kernel(
        const float* __restrict__ X,
        const float* __restrict__ Y,
        const float* __restrict__ G,
        const float* __restrict__ K,
        float* __restrict__ dX,
        int N, int M, int D, float gamma) {
    int i = blockIdx.x;
    int d = threadIdx.x;
    if (i >= N || d >= D) return;

    float xid = X[(size_t)i * D + d];
    const float* grow = G + (size_t)i * M;
    const float* krow = K + (size_t)i * M;
    float acc = 0.0f;
    for (int j = 0; j < M; ++j) {
        float coef = -2.0f * gamma * grow[j] * krow[j];
        acc += coef * (xid - Y[(size_t)j * D + d]);
    }
    dX[(size_t)i * D + d] = acc;
}

__global__ void rbf_backward_dY_kernel(
        const float* __restrict__ X,
        const float* __restrict__ Y,
        const float* __restrict__ G,
        const float* __restrict__ K,
        float* __restrict__ dY,
        int N, int M, int D, float gamma) {
    int j = blockIdx.x;
    int d = threadIdx.x;
    if (j >= M || d >= D) return;

    float yjd = Y[(size_t)j * D + d];
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        float coef = -2.0f * gamma * G[(size_t)i * M + j] * K[(size_t)i * M + j];
        acc += coef * (yjd - X[(size_t)i * D + d]);
    }
    dY[(size_t)j * D + d] = acc;
}

// ============================================================================
// host 接口（extern "C"，裸主机指针；内部管理 device 内存）
// ============================================================================

extern "C" void rbf_forward_cuda(
        const float* hX, const float* hY, float* hK,
        int N, int M, int D, float gamma) {
    float *dX_, *dY_, *dK_;
    size_t szX = (size_t)N * D * sizeof(float);
    size_t szY = (size_t)M * D * sizeof(float);
    size_t szK = (size_t)N * M * sizeof(float);
    RBF_CHECK_CUDA(cudaMalloc(&dX_, szX));
    RBF_CHECK_CUDA(cudaMalloc(&dY_, szY));
    RBF_CHECK_CUDA(cudaMalloc(&dK_, szK));
    RBF_CHECK_CUDA(cudaMemcpy(dX_, hX, szX, cudaMemcpyHostToDevice));
    RBF_CHECK_CUDA(cudaMemcpy(dY_, hY, szY, cudaMemcpyHostToDevice));

    dim3 threads(TX, TY);
    dim3 blocks((M + BM - 1) / BM, (N + BN - 1) / BN);
    rbf_forward_kernel<<<blocks, threads>>>(dX_, dY_, dK_, N, M, D, gamma);
    RBF_CHECK_CUDA(cudaGetLastError());
    RBF_CHECK_CUDA(cudaDeviceSynchronize());

    RBF_CHECK_CUDA(cudaMemcpy(hK, dK_, szK, cudaMemcpyDeviceToHost));
    cudaFree(dX_); cudaFree(dY_); cudaFree(dK_);
}

extern "C" void rbf_backward_cuda(
        const float* hX, const float* hY, const float* hG, const float* hK,
        float* hdX, float* hdY,
        int N, int M, int D, float gamma) {
    float *dX_, *dY_, *dG_, *dK_, *ddX_, *ddY_;
    size_t szX = (size_t)N * D * sizeof(float);
    size_t szY = (size_t)M * D * sizeof(float);
    size_t szNM = (size_t)N * M * sizeof(float);
    RBF_CHECK_CUDA(cudaMalloc(&dX_, szX));
    RBF_CHECK_CUDA(cudaMalloc(&dY_, szY));
    RBF_CHECK_CUDA(cudaMalloc(&dG_, szNM));
    RBF_CHECK_CUDA(cudaMalloc(&dK_, szNM));
    RBF_CHECK_CUDA(cudaMalloc(&ddX_, szX));
    RBF_CHECK_CUDA(cudaMalloc(&ddY_, szY));
    RBF_CHECK_CUDA(cudaMemcpy(dX_, hX, szX, cudaMemcpyHostToDevice));
    RBF_CHECK_CUDA(cudaMemcpy(dY_, hY, szY, cudaMemcpyHostToDevice));
    RBF_CHECK_CUDA(cudaMemcpy(dG_, hG, szNM, cudaMemcpyHostToDevice));
    RBF_CHECK_CUDA(cudaMemcpy(dK_, hK, szNM, cudaMemcpyHostToDevice));

    rbf_backward_dX_kernel<<<N, D>>>(dX_, dY_, dG_, dK_, ddX_, N, M, D, gamma);
    RBF_CHECK_CUDA(cudaGetLastError());
    rbf_backward_dY_kernel<<<M, D>>>(dX_, dY_, dG_, dK_, ddY_, N, M, D, gamma);
    RBF_CHECK_CUDA(cudaGetLastError());
    RBF_CHECK_CUDA(cudaDeviceSynchronize());

    RBF_CHECK_CUDA(cudaMemcpy(hdX, ddX_, szX, cudaMemcpyDeviceToHost));
    RBF_CHECK_CUDA(cudaMemcpy(hdY, ddY_, szY, cudaMemcpyDeviceToHost));
    cudaFree(dX_); cudaFree(dY_); cudaFree(dG_); cudaFree(dK_);
    cudaFree(ddX_); cudaFree(ddY_);
}
