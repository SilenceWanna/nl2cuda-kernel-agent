// RBF 高斯核矩阵——前向 kernel（阶段1：朴素正确版，不追求速度）。
//   X:[N,D], Y:[M,D]
//   Dist[i,j] = sum_d (X[i,d] - Y[j,d])^2
//   K[i,j]    = exp(-gamma * Dist[i,j])
//
// 朴素策略：一个线程算一个输出元素 K[i,j]，内部循环 D 维累加平方差。
// 每个输出独立，无中间张量物化（这正是相对广播参考的内存优势来源）。
// 优化（tiling / shared memory / float4 等）留到阶段3。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rbf_forward_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        float* __restrict__ K,         // [N, M]
        int N, int M, int D, float gamma) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 行 index (0..N)
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 列 index (0..M)
    if (i >= N || j >= M) return;

    const float* xi = X + (size_t)i * D;
    const float* yj = Y + (size_t)j * D;

    float dist = 0.0f;
    for (int d = 0; d < D; ++d) {
        float diff = xi[d] - yj[d];
        dist += diff * diff;
    }
    K[(size_t)i * M + j] = expf(-gamma * dist);
}

// 前向：X:[N,D], Y:[M,D] (float32, CUDA) -> K:[N,M]
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

    dim3 threads(16, 16);
    dim3 blocks((M + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);
    rbf_forward_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), K.data_ptr<float>(),
        N, M, D, (float)gamma);

    return K;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbf_forward", &rbf_forward, "RBF kernel matrix forward (CUDA, naive)");
}
