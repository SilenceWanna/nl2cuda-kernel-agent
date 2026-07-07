// RBF 反向 kernel（阶段3优化版：block-per-row，高 occupancy）。
//
// 前向 K[i,j]=exp(-gamma*dist_ij)；S[i,j]=-gamma*G[i,j]*K[i,j]；
//   dX[i,d] = sum_j S[i,j] * 2*(X[i,d]-Y[j,d])
//   dY[j,d] = sum_i S[i,j] * 2*(Y[j,d]-X[i,d])
//
// 朴素版一线程算一整行 → 仅 N(或 M) 个线程，occupancy 极低（47ms 主因）。
// 本版：一个 block 处理一行，blockDim=D 个线程各管一个分量 d；
//   block 内协作规约求 dist_ij（shared），S 由所有线程共享；
//   每线程只维护 1 个寄存器累加器 acc（该行的 dX[i,d] 或 dY[j,d]）。
//   grid = N(或 M) 个 block，每 SM 可驻留多个 block → occupancy 高。
// 保持 fp32、精确 expf。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// dX：block = 行 i，thread = 分量 d
__global__ void rbf_backward_dX_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        const float* __restrict__ G,   // [N, M]
        float* __restrict__ dX,        // [N, D]
        int N, int M, int D, float gamma) {
    int i = blockIdx.x;
    int d = threadIdx.x;
    if (i >= N || d >= D) return;

    extern __shared__ float red[];     // blockDim.x 个，用于 dist 规约

    float xid = X[(size_t)i * D + d];  // 该行分量，只读一次
    const float* grow = G + (size_t)i * M;
    float acc = 0.0f;

    for (int j = 0; j < M; ++j) {
        float yjd = Y[(size_t)j * D + d];
        float diff = xid - yjd;
        // block 内规约 diff^2 -> dist_ij
        red[d] = diff * diff;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (d < s) red[d] += red[d + s];
            __syncthreads();
        }
        float dist = red[0];
        __syncthreads();
        float K = expf(-gamma * dist);
        float S = -gamma * grow[j] * K;
        acc += 2.0f * S * diff;        // diff = xid - yjd
    }
    dX[(size_t)i * D + d] = acc;
}

// dY：block = 行 j，thread = 分量 d
__global__ void rbf_backward_dY_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        const float* __restrict__ G,   // [N, M]
        float* __restrict__ dY,        // [M, D]
        int N, int M, int D, float gamma) {
    int j = blockIdx.x;
    int d = threadIdx.x;
    if (j >= M || d >= D) return;

    extern __shared__ float red[];

    float yjd = Y[(size_t)j * D + d];
    float acc = 0.0f;

    for (int i = 0; i < N; ++i) {
        float xid = X[(size_t)i * D + d];
        float diff = yjd - xid;
        red[d] = diff * diff;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (d < s) red[d] += red[d + s];
            __syncthreads();
        }
        float dist = red[0];
        __syncthreads();
        float K = expf(-gamma * dist);
        float S = -gamma * G[(size_t)i * M + j] * K;
        acc += 2.0f * S * diff;        // diff = yjd - xid
    }
    dY[(size_t)j * D + d] = acc;
}

std::vector<torch::Tensor> rbf_backward(
        torch::Tensor X, torch::Tensor Y, torch::Tensor G, double gamma) {
    TORCH_CHECK(X.is_cuda() && Y.is_cuda() && G.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && Y.dtype() == torch::kFloat32
                && G.dtype() == torch::kFloat32, "inputs must be float32");
    X = X.contiguous();
    Y = Y.contiguous();
    G = G.contiguous();
    int N = X.size(0);
    int M = Y.size(0);
    int D = X.size(1);

    auto dX = torch::empty_like(X);
    auto dY = torch::empty_like(Y);

    // 规约 kernel 要求 blockDim = D 为 2 的幂（本任务 D=64 满足）
    size_t shmem = D * sizeof(float);
    rbf_backward_dX_kernel<<<N, D, shmem>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), G.data_ptr<float>(),
        dX.data_ptr<float>(), N, M, D, (float)gamma);
    rbf_backward_dY_kernel<<<M, D, shmem>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), G.data_ptr<float>(),
        dY.data_ptr<float>(), N, M, D, (float)gamma);

    return {dX, dY};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbf_backward", &rbf_backward, "RBF kernel matrix backward (CUDA, block-per-row)");
}
