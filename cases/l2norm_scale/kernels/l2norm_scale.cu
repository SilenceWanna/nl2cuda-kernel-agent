// l2norm_scale 前向 + 反向 CUDA kernel（fp32, float4 向量化, 一 block 一行）。
// 前向: norm[i]=sqrt(sum_j x^2+eps); Y[i,j]=x[i,j]/norm[i]*g[j]; 缓存 norm 供反向复用。
// 反向: dg[j]=sum_i G[i,j]*xhat[i,j]; dX[i]=(g*G[i]-xhat[i]*dot_i)/norm[i], dot_i=sum_j xhat[i,j]*g[j]*G[i,j]。
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#ifndef L2N_THREADS
#define L2N_THREADS 256
#endif

__device__ __forceinline__ float block_reduce_sum(float v, float* sh) {
    int t = threadIdx.x;
    // warp 内规约
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if ((t & 31) == 0) sh[t >> 5] = v;
    __syncthreads();
    // 首 warp 汇总各 warp 部分和
    int nwarp = blockDim.x >> 5;
    v = (t < nwarp) ? sh[t] : 0.0f;
    if (t < 32) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    }
    if (t == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// ---- 前向（一 block 一行；float4。前向读写量=baseline 属准带宽墙——chunk+g缓存、寄存器缓存X 实测均无益，用最简 float4 版）----
__global__ void l2n_forward_kernel(
    const float* __restrict__ X, const float* __restrict__ g,
    float* __restrict__ Y, float* __restrict__ norm_out,
    int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = X + (long)row * D;
    float* yr = Y + (long)row * D;
    __shared__ float sh[32];
    if ((D & 3) == 0) {
        const float4* xr4 = reinterpret_cast<const float4*>(xr);
        const float4* g4  = reinterpret_cast<const float4*>(g);
        float4* yr4       = reinterpret_cast<float4*>(yr);
        int D4 = D >> 2;
        float part = 0.0f;
        for (int j = threadIdx.x; j < D4; j += blockDim.x) {
            float4 x = xr4[j];
            part += x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w;
        }
        float norm = sqrtf(block_reduce_sum(part, sh) + eps);
        if (threadIdx.x == 0) norm_out[row] = norm;
        float inv = 1.0f / norm;
        for (int j = threadIdx.x; j < D4; j += blockDim.x) {
            float4 x = xr4[j], gg = g4[j], y;
            y.x = x.x*inv*gg.x; y.y = x.y*inv*gg.y;
            y.z = x.z*inv*gg.z; y.w = x.w*inv*gg.w;
            yr4[j] = y;
        }
        return;
    }
    float part = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) part += xr[j]*xr[j];
    float norm = sqrtf(block_reduce_sum(part, sh) + eps);
    if (threadIdx.x == 0) norm_out[row] = norm;
    float inv = 1.0f / norm;
    for (int j = threadIdx.x; j < D; j += blockDim.x) yr[j] = xr[j] * inv * g[j];
}

// ---- 反向融合（一 block 管一段行 chunk）：X/G 只读一遍，同时算 dX 写出 +
//      shared 私有累积本 chunk 对 dg 的列贡献，chunk 结束一次性 atomicAdd 到全局 dg。
//      atomic 次数 = 行块数（≪N），既省 dg 的整遍重扫、又避 per-element atomic 竞争。----
#ifndef L2N_ROWS_PER_BLK
#define L2N_ROWS_PER_BLK 32
#endif
__global__ void l2n_backward_fused_kernel(
    const float* __restrict__ X, const float* __restrict__ g,
    const float* __restrict__ G, const float* __restrict__ norm_in,
    float* __restrict__ dX, float* __restrict__ dg,
    int N, int D) {
    extern __shared__ float dg_part[];   // [D] 本 block 的 dg 部分和
    __shared__ float sh[32];
    for (int c = threadIdx.x; c < D; c += blockDim.x) dg_part[c] = 0.0f;
    __syncthreads();

    int row0 = blockIdx.x * L2N_ROWS_PER_BLK;
    int row1 = min(row0 + L2N_ROWS_PER_BLK, N);
    int D4 = D >> 2;
    bool vec = ((D & 3) == 0);

    for (int row = row0; row < row1; ++row) {
        const float* xr = X + (long)row * D;
        const float* gr = G + (long)row * D;
        float* dxr = dX + (long)row * D;
        float inv = 1.0f / norm_in[row];
        // dot_i = Σ_j xhat·g·G（行内规约）
        float part = 0.0f;
        if (vec) {
            const float4* xr4 = reinterpret_cast<const float4*>(xr);
            const float4* gr4 = reinterpret_cast<const float4*>(gr);
            const float4* g4  = reinterpret_cast<const float4*>(g);
            for (int j = threadIdx.x; j < D4; j += blockDim.x) {
                float4 x = xr4[j], gg = g4[j], gd = gr4[j];
                part += inv*(x.x*gg.x*gd.x + x.y*gg.y*gd.y + x.z*gg.z*gd.z + x.w*gg.w*gd.w);
            }
        } else {
            for (int j = threadIdx.x; j < D; j += blockDim.x)
                part += xr[j]*inv * g[j] * gr[j];
        }
        float dot = block_reduce_sum(part, sh);
        // 写 dX + 累积 dg_part（复用本行已读的 X/G）
        if (vec) {
            const float4* xr4 = reinterpret_cast<const float4*>(xr);
            const float4* gr4 = reinterpret_cast<const float4*>(gr);
            const float4* g4  = reinterpret_cast<const float4*>(g);
            float4* dxr4      = reinterpret_cast<float4*>(dxr);
            for (int j = threadIdx.x; j < D4; j += blockDim.x) {
                float4 x = xr4[j], gg = g4[j], gd = gr4[j], d;
                float xh;
                d.x = (gg.x*gd.x - x.x*inv*dot)*inv; xh = x.x*inv; dg_part[(j<<2)]   += gd.x*xh;
                d.y = (gg.y*gd.y - x.y*inv*dot)*inv; xh = x.y*inv; dg_part[(j<<2)+1] += gd.y*xh;
                d.z = (gg.z*gd.z - x.z*inv*dot)*inv; xh = x.z*inv; dg_part[(j<<2)+2] += gd.z*xh;
                d.w = (gg.w*gd.w - x.w*inv*dot)*inv; xh = x.w*inv; dg_part[(j<<2)+3] += gd.w*xh;
                dxr4[j] = d;
            }
        } else {
            for (int j = threadIdx.x; j < D; j += blockDim.x) {
                float xh = xr[j]*inv;
                dxr[j] = (g[j]*gr[j] - xh*dot)*inv;
                dg_part[j] += gr[j]*xh;
            }
        }
        __syncthreads();   // dg_part 跨行累积前确保本行写完（同线程负责同列，无需，但 dot 用 sh 需同步）
    }
    // chunk 结束：dg_part 一次性 atomicAdd 到全局 dg（每列 atomic 次数=行块数）
    for (int c = threadIdx.x; c < D; c += blockDim.x) atomicAdd(&dg[c], dg_part[c]);
}

// ---- host 绑定 ----
std::vector<torch::Tensor> l2n_forward(torch::Tensor X, torch::Tensor g, double eps) {
    int N = X.size(0), D = X.size(1);
    auto Y = torch::empty_like(X);
    auto norm = torch::empty({N}, X.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    l2n_forward_kernel<<<N, L2N_THREADS, 0, stream>>>(
        X.data_ptr<float>(), g.data_ptr<float>(),
        Y.data_ptr<float>(), norm.data_ptr<float>(), N, D, (float)eps);
    return {Y, norm};
}

std::vector<torch::Tensor> l2n_backward(
    torch::Tensor X, torch::Tensor g, torch::Tensor G, torch::Tensor norm) {
    int N = X.size(0), D = X.size(1);
    auto dX = torch::empty_like(X);
    auto dg = torch::zeros({D}, X.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    // 融合 kernel：一 block 管一段行，X/G 只读一遍算 dX + shared 累积 dg，chunk 末一次 atomicAdd
    int row_blocks = (N + L2N_ROWS_PER_BLK - 1) / L2N_ROWS_PER_BLK;
    size_t shmem = (size_t)D * sizeof(float);
    l2n_backward_fused_kernel<<<row_blocks, L2N_THREADS, shmem, stream>>>(
        X.data_ptr<float>(), g.data_ptr<float>(), G.data_ptr<float>(),
        norm.data_ptr<float>(), dX.data_ptr<float>(), dg.data_ptr<float>(), N, D);
    return {dX, dg};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l2n_forward", &l2n_forward, "L2norm-scale forward");
    m.def("l2n_backward", &l2n_backward, "L2norm-scale backward");
}
