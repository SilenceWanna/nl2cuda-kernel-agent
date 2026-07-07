// RBF 高斯核矩阵——反向 kernel（阶段1：朴素正确版，不追求速度）。
//
// 前向： K[i,j] = exp(-gamma * dist_ij),  dist_ij = sum_d (X[i,d]-Y[j,d])^2
// 给定上游梯度 G[i,j] = dL/dK[i,j]，记 S[i,j] = -gamma * G[i,j] * K[i,j]，则：
//   dX[i,d] = sum_j S[i,j] * 2*(X[i,d]-Y[j,d])
//   dY[j,d] = sum_i S[i,j] * 2*(Y[j,d]-X[i,d])
//
// 朴素策略：
//   dX kernel：一个线程算一行 dX[i,:]，沿 j 归约（重算 dist/K，不依赖前向缓存）。
//   dY kernel：一个线程算一行 dY[j,:]，沿 i 归约。
// 每行内对 D 维用局部数组累加。D=64，用寄存器数组即可。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_D 128  // 局部累加数组上限（D<=128 足够覆盖本任务 D=64）

// dX[i,:] = sum_j (-gamma * G[i,j] * K[i,j]) * 2*(X[i,d]-Y[j,d])
__global__ void rbf_backward_dX_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        const float* __restrict__ G,   // [N, M] upstream grad
        float* __restrict__ dX,        // [N, D]
        int N, int M, int D, float gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* xi = X + (size_t)i * D;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;

    for (int j = 0; j < M; ++j) {
        const float* yj = Y + (size_t)j * D;
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = xi[d] - yj[d];
            dist += diff * diff;
        }
        float K = expf(-gamma * dist);
        float S = -gamma * G[(size_t)i * M + j] * K;   // dL/d(dist_ij)
        float coef = 2.0f * S;
        for (int d = 0; d < D; ++d) {
            acc[d] += coef * (xi[d] - yj[d]);
        }
    }
    float* out = dX + (size_t)i * D;
    for (int d = 0; d < D; ++d) out[d] = acc[d];
}

// dY[j,:] = sum_i (-gamma * G[i,j] * K[i,j]) * 2*(Y[j,d]-X[i,d])
__global__ void rbf_backward_dY_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        const float* __restrict__ G,   // [N, M]
        float* __restrict__ dY,        // [M, D]
        int N, int M, int D, float gamma) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;

    const float* yj = Y + (size_t)j * D;
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;

    for (int i = 0; i < N; ++i) {
        const float* xi = X + (size_t)i * D;
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = xi[d] - yj[d];
            dist += diff * diff;
        }
        float K = expf(-gamma * dist);
        float S = -gamma * G[(size_t)i * M + j] * K;
        float coef = 2.0f * S;
        for (int d = 0; d < D; ++d) {
            acc[d] += coef * (yj[d] - xi[d]);
        }
    }
    float* out = dY + (size_t)j * D;
    for (int d = 0; d < D; ++d) out[d] = acc[d];
}

// 反向：给定 X, Y, G(=dL/dK) -> (dX, dY)
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
    TORCH_CHECK(D <= MAX_D, "D exceeds MAX_D");

    auto dX = torch::empty_like(X);
    auto dY = torch::empty_like(Y);

    const int threads = 128;
    rbf_backward_dX_kernel<<<(N + threads - 1) / threads, threads>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), G.data_ptr<float>(),
        dX.data_ptr<float>(), N, M, D, (float)gamma);
    rbf_backward_dY_kernel<<<(M + threads - 1) / threads, threads>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), G.data_ptr<float>(),
        dY.data_ptr<float>(), N, M, D, (float)gamma);

    return {dX, dY};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbf_backward", &rbf_backward, "RBF kernel matrix backward (CUDA, naive)");
}
