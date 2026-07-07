// RBF 反向 kernel（阶段3优化版 v2：前向缓存 K 复用，消除 dist/exp 重算 + per-j 规约）。
//
// 关键优化（loop.md #2 前向缓存复用）：前向已算出 K[i,j]，autograd 保存后传入反向，
// 则反向无需重算 dist/exp，且每个 j 的贡献系数 coef[i,j]=-2γ·G[i,j]·K[i,j] 是标量，
// 各分量 d 独立累加，**不再需要 per-j 的 block 内规约**（同步开销归零）。
//   dX[i,d] = sum_j coef[i,j] * (X[i,d]-Y[j,d])
//   dY[j,d] = sum_i coef[i,j] * (Y[j,d]-X[i,d]) = -sum_i coef[i,j]*(X[i,d]-Y[j,d])
//
// block-per-row，thread-per-d，高 occupancy；保持 fp32。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// dX：block = 行 i，thread = 分量 d
__global__ void rbf_backward_dX_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        const float* __restrict__ G,   // [N, M]
        const float* __restrict__ K,   // [N, M] 前向缓存
        float* __restrict__ dX,        // [N, D]
        int N, int M, int D, float gamma) {
    int i = blockIdx.x;
    int d = threadIdx.x;
    if (i >= N || d >= D) return;

    float xid = X[(size_t)i * D + d];
    const float* grow = G + (size_t)i * M;
    const float* krow = K + (size_t)i * M;
    float acc = 0.0f;
    for (int j = 0; j < M; ++j) {
        float coef = -2.0f * gamma * grow[j] * krow[j];   // 标量，无需规约
        acc += coef * (xid - Y[(size_t)j * D + d]);
    }
    dX[(size_t)i * D + d] = acc;
}

// dY：block = 行 j，thread = 分量 d
__global__ void rbf_backward_dY_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        const float* __restrict__ G,   // [N, M]
        const float* __restrict__ K,   // [N, M]
        float* __restrict__ dY,        // [M, D]
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

std::vector<torch::Tensor> rbf_backward(
        torch::Tensor X, torch::Tensor Y, torch::Tensor G, torch::Tensor K, double gamma) {
    TORCH_CHECK(X.is_cuda() && Y.is_cuda() && G.is_cuda() && K.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && Y.dtype() == torch::kFloat32
                && G.dtype() == torch::kFloat32 && K.dtype() == torch::kFloat32,
                "inputs must be float32");
    X = X.contiguous();
    Y = Y.contiguous();
    G = G.contiguous();
    K = K.contiguous();
    int N = X.size(0);
    int M = Y.size(0);
    int D = X.size(1);

    auto dX = torch::empty_like(X);
    auto dY = torch::empty_like(Y);

    rbf_backward_dX_kernel<<<N, D>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), G.data_ptr<float>(), K.data_ptr<float>(),
        dX.data_ptr<float>(), N, M, D, (float)gamma);
    rbf_backward_dY_kernel<<<M, D>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), G.data_ptr<float>(), K.data_ptr<float>(),
        dY.data_ptr<float>(), N, M, D, (float)gamma);

    return {dX, dY};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbf_backward", &rbf_backward, "RBF kernel matrix backward (CUDA, cached-K)");
}
