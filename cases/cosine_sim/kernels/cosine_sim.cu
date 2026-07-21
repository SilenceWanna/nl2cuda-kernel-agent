// cosine_sim 前反向 CUDA kernel（第1轮：预计算单位化向量+范数并缓存复用，消除重算）。
// 前向: Ah=A/|A|, Bh=B/|B| (各 block-per-row 单位化)；S[i,j]=Ah[i]·Bh[j]。
// 反向: dAh[i]=Σ_j dS[i,j]·Bh[j]; dA[i]=(dAh[i]-(Ah[i]·dAh[i])·Ah[i])·invNormA[i]; dB 对称。
//   —— 反向复用缓存的 Ah/Bh/invNorm，不再重算 norm/dot（消除朴素版 O(N·D·M·D) 重算）。
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define THREADS 256

__device__ __forceinline__ float blk_reduce(float v, float* sh) {
    int t = threadIdx.x;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if ((t & 31) == 0) sh[t >> 5] = v;
    __syncthreads();
    int nw = blockDim.x >> 5;
    v = (t < nw) ? sh[t] : 0.f;
    if (t < 32) for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (t == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// 单位化：一 block 一行，输出 Rh=R/|R|，并存 invNorm=1/(|R|+eps)
__global__ void normalize_kernel(
    const float* __restrict__ R, float* __restrict__ Rh, float* __restrict__ invNorm,
    int rows, int D, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* rr = R + (long)r * D;
    float* oh = Rh + (long)r * D;
    __shared__ float sh[32];
    float part = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) part += rr[d] * rr[d];
    float ss = blk_reduce(part, sh);
    float inv = 1.f / (sqrtf(ss) + eps);
    if (threadIdx.x == 0) invNorm[r] = inv;
    for (int d = threadIdx.x; d < D; d += blockDim.x) oh[d] = rr[d] * inv;
}

// 前向 S：tiling，一 block 算一片 [BM×BN] 的 S，协作载入 Ah/Bh tile 到 shared
#define TILE 16
__global__ void cos_fwd_tiled(
    const float* __restrict__ Ah, const float* __restrict__ Bh,
    float* __restrict__ S, int N, int M, int D) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int d0 = 0; d0 < D; d0 += TILE) {
        int di = d0 + threadIdx.x, dj = d0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (i < N && di < D) ? Ah[(long)i * D + di] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (j < M && dj < D) ? Bh[(long)j * D + dj] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (i < N && j < M) S[(long)i * M + j] = acc;
}

// 反向 dA：一 block 一行 i，blockDim=D，每线程 d 私有累加 dAh[d]=Σ_j dS[i,j]·Bh[j,d]（寄存器，无 atomic）。
__global__ void cos_bwd_dA(
    const float* __restrict__ Ah, const float* __restrict__ Bh,
    const float* __restrict__ dS, const float* __restrict__ invNormA,
    float* __restrict__ dA, int N, int M, int D) {
    int i = blockIdx.x;
    int d = threadIdx.x;
    if (i >= N || d >= D) return;
    const float* dsr = dS + (long)i * M;
    float acc = 0.f;                                  // 本线程负责的 dAh[d]，寄存器累加
    for (int j = 0; j < M; ++j) acc += dsr[j] * Bh[(long)j * D + d];
    __shared__ float dah[1024];                       // 本行 dAh（D<=1024）
    dah[d] = acc;
    __syncthreads();
    __shared__ float sh[32];
    const float* ah = Ah + (long)i * D;
    float part = ah[d] * dah[d];                       // proj = Ah[i]·dAh[i]
    part = blk_reduce(part, sh);
    dA[(long)i * D + d] = (acc - part * ah[d]) * invNormA[i];
}

// 反向 dB：对称，一 block 一行 j，dBh[d]=Σ_i dS[i,j]·Ah[i,d]。
__global__ void cos_bwd_dB(
    const float* __restrict__ Ah, const float* __restrict__ Bh,
    const float* __restrict__ dS, const float* __restrict__ invNormB,
    float* __restrict__ dB, int N, int M, int D) {
    int j = blockIdx.x;
    int d = threadIdx.x;
    if (j >= M || d >= D) return;
    float acc = 0.f;
    for (int i = 0; i < N; ++i) acc += dS[(long)i * M + j] * Ah[(long)i * D + d];
    __shared__ float dbh[1024];
    dbh[d] = acc;
    __syncthreads();
    __shared__ float sh[32];
    const float* bh = Bh + (long)j * D;
    float part = bh[d] * dbh[d];
    part = blk_reduce(part, sh);
    dB[(long)j * D + d] = (acc - part * bh[d]) * invNormB[j];
}

// ---- host ----
std::vector<torch::Tensor> cos_forward(torch::Tensor A, torch::Tensor B, double eps) {
    int N = A.size(0), M = B.size(0), D = A.size(1);
    auto Ah = torch::empty_like(A), Bh = torch::empty_like(B);
    auto invA = torch::empty({N}, A.options()), invB = torch::empty({M}, B.options());
    auto S = torch::empty({N, M}, A.options());
    auto st = c10::cuda::getCurrentCUDAStream();
    normalize_kernel<<<N, THREADS, 0, st>>>(A.data_ptr<float>(), Ah.data_ptr<float>(), invA.data_ptr<float>(), N, D, (float)eps);
    normalize_kernel<<<M, THREADS, 0, st>>>(B.data_ptr<float>(), Bh.data_ptr<float>(), invB.data_ptr<float>(), M, D, (float)eps);
    dim3 blk(TILE, TILE), grd((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    cos_fwd_tiled<<<grd, blk, 0, st>>>(Ah.data_ptr<float>(), Bh.data_ptr<float>(), S.data_ptr<float>(), N, M, D);
    return {S, Ah, Bh, invA, invB};
}

std::vector<torch::Tensor> cos_backward(
    torch::Tensor Ah, torch::Tensor Bh, torch::Tensor invA, torch::Tensor invB,
    torch::Tensor dS, int N, int M, int D) {
    auto dA = torch::empty({N, D}, Ah.options()), dB = torch::empty({M, D}, Bh.options());
    auto st = c10::cuda::getCurrentCUDAStream();
    cos_bwd_dA<<<N, D, 0, st>>>(Ah.data_ptr<float>(), Bh.data_ptr<float>(), dS.data_ptr<float>(), invA.data_ptr<float>(), dA.data_ptr<float>(), N, M, D);
    cos_bwd_dB<<<M, D, 0, st>>>(Ah.data_ptr<float>(), Bh.data_ptr<float>(), dS.data_ptr<float>(), invB.data_ptr<float>(), dB.data_ptr<float>(), N, M, D);
    return {dA, dB};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cos_forward", &cos_forward, "cosine sim forward (cached norms)");
    m.def("cos_backward", &cos_backward, "cosine sim backward (reuse cached)");
}
