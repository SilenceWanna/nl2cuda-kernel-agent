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

// ---- 前向 ----
__global__ void l2n_forward_kernel(
    const float* __restrict__ X, const float* __restrict__ g,
    float* __restrict__ Y, float* __restrict__ norm_out,
    int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = X + (long)row * D;
    float* yr = Y + (long)row * D;
    __shared__ float sh[32];

    // D%4==0 走 float4 快路径：每线程读 float4 存寄存器，Σx² 与写 Y 复用（X 只读一遍）
    if ((D & 3) == 0) {
        const float4* xr4 = reinterpret_cast<const float4*>(xr);
        const float4* g4  = reinterpret_cast<const float4*>(g);
        float4* yr4       = reinterpret_cast<float4*>(yr);
        int D4 = D >> 2;
        float part = 0.0f;
        // 缓存本线程负责的 float4（跨步），寄存器复用
        for (int j = threadIdx.x; j < D4; j += blockDim.x) {
            float4 x = xr4[j];
            part += x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w;
        }
        float ssq = block_reduce_sum(part, sh);
        float norm = sqrtf(ssq + eps);
        if (threadIdx.x == 0) norm_out[row] = norm;
        float inv = 1.0f / norm;
        for (int j = threadIdx.x; j < D4; j += blockDim.x) {
            float4 x = xr4[j]; float4 gg = g4[j]; float4 y;
            y.x = x.x*inv*gg.x; y.y = x.y*inv*gg.y;
            y.z = x.z*inv*gg.z; y.w = x.w*inv*gg.w;
            yr4[j] = y;
        }
        return;
    }
    // 通用回退
    float part = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float x = xr[j];
        part += x * x;
    }
    float ssq = block_reduce_sum(part, sh);
    float norm = sqrtf(ssq + eps);
    if (threadIdx.x == 0) norm_out[row] = norm;
    float inv = 1.0f / norm;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        yr[j] = xr[j] * inv * g[j];
    }
}

// ---- 反向：dX（一 block 一行，规约 dot_i=sum xhat*g*G）----
__global__ void l2n_backward_dx_kernel(
    const float* __restrict__ X, const float* __restrict__ g,
    const float* __restrict__ G, const float* __restrict__ norm_in,
    float* __restrict__ dX,
    int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = X + (long)row * D;
    const float* gr = G + (long)row * D;
    float* dxr = dX + (long)row * D;
    __shared__ float sh[32];
    float norm = norm_in[row];
    float inv = 1.0f / norm;

    // float4 快路径：dot=Σ xhat*g*G 与 dX 写出（各读一遍 float4，避免寄存器缓存压 occupancy）
    if ((D & 3) == 0) {
        const float4* xr4 = reinterpret_cast<const float4*>(xr);
        const float4* gr4 = reinterpret_cast<const float4*>(gr);
        const float4* g4  = reinterpret_cast<const float4*>(g);
        float4* dxr4      = reinterpret_cast<float4*>(dxr);
        int D4 = D >> 2;
        float part = 0.0f;
        for (int j = threadIdx.x; j < D4; j += blockDim.x) {
            float4 x = xr4[j], gg = g4[j], gr = gr4[j];
            part += inv*(x.x*gg.x*gr.x + x.y*gg.y*gr.y + x.z*gg.z*gr.z + x.w*gg.w*gr.w);
        }
        float dot = block_reduce_sum(part, sh);
        for (int j = threadIdx.x; j < D4; j += blockDim.x) {
            float4 x = xr4[j], gg = g4[j], gr = gr4[j], d;
            d.x = (gg.x*gr.x - x.x*inv*dot)*inv;
            d.y = (gg.y*gr.y - x.y*inv*dot)*inv;
            d.z = (gg.z*gr.z - x.z*inv*dot)*inv;
            d.w = (gg.w*gr.w - x.w*inv*dot)*inv;
            dxr4[j] = d;
        }
        return;
    }
    // 通用回退
    float part = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float xhat = xr[j] * inv;
        part += xhat * g[j] * gr[j];
    }
    float dot = block_reduce_sum(part, sh);
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float xhat = xr[j] * inv;
        dxr[j] = (g[j] * gr[j] - xhat * dot) * inv;
    }
}

// ---- 反向：dg = sum_i G[i,j]*xhat[i,j]（跨行列规约，2D 分块 + atomicAdd）----
// 每 block 处理一段行 × 一段连续列，块内沿行累加，再 atomicAdd 到全局 dg。
__global__ void l2n_backward_dg_kernel(
    const float* __restrict__ X, const float* __restrict__ G,
    const float* __restrict__ norm_in, float* __restrict__ dg,
    int N, int D) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    if (col >= D) return;
    float acc = 0.0f;
    // 沿行分块：blockIdx.y 步进
    for (int row = blockIdx.y; row < N; row += gridDim.y) {
        float inv = 1.0f / norm_in[row];
        float xhat = X[(long)row * D + col] * inv;
        acc += G[(long)row * D + col] * xhat;
    }
    atomicAdd(&dg[col], acc);
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

    l2n_backward_dx_kernel<<<N, L2N_THREADS, 0, stream>>>(
        X.data_ptr<float>(), g.data_ptr<float>(), G.data_ptr<float>(),
        norm.data_ptr<float>(), dX.data_ptr<float>(), N, D);

    int col_threads = 256;
    int col_blocks = (D + col_threads - 1) / col_threads;
    int row_blocks = 256;  // 沿行分块并行度（256 是并行度与 atomic 竞争的较优平衡点）
    dim3 grid(col_blocks, row_blocks);
    l2n_backward_dg_kernel<<<grid, col_threads, 0, stream>>>(
        X.data_ptr<float>(), G.data_ptr<float>(), norm.data_ptr<float>(),
        dg.data_ptr<float>(), N, D);

    return {dX, dg};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l2n_forward", &l2n_forward, "L2norm-scale forward");
    m.def("l2n_backward", &l2n_backward, "L2norm-scale backward");
}
