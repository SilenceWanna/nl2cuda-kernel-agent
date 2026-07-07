// RBF 前向 kernel（阶段3 v3：极简高占用率，无 shared，靠 L2 缓存）。
//   诊断实验：D=64 时 X+Y 仅 ~1MB，可被 T4 4MB L2 全部吸收，反复读命中 L2。
//   猜想：tiling 的 shared 同步/寄存器开销对这个小问题得不偿失，去掉反而更快。
//   一个线程算一个 K[i,j]，256 线程/block（32×8），float4 读全局。保持 fp32。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rbf_forward_kernel(
        const float* __restrict__ X,   // [N, D]
        const float* __restrict__ Y,   // [M, D]
        float* __restrict__ K,         // [N, M]
        int N, int M, int D, float gamma) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 列 (Y 行)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 行 (X 行)
    if (i >= N || j >= M) return;

    const float4* xi = reinterpret_cast<const float4*>(X + (size_t)i * D);
    const float4* yj = reinterpret_cast<const float4*>(Y + (size_t)j * D);
    int D4 = D >> 2;

    float dist = 0.0f;
    #pragma unroll 4
    for (int q = 0; q < D4; ++q) {
        float4 xv = xi[q], yv = yj[q];
        float a = xv.x - yv.x, b = xv.y - yv.y, c = xv.z - yv.z, e = xv.w - yv.w;
        dist += a*a + b*b + c*c + e*e;
    }
    K[(size_t)i * M + j] = expf(-gamma * dist);
}

torch::Tensor rbf_forward(torch::Tensor X, torch::Tensor Y, double gamma) {
    TORCH_CHECK(X.is_cuda() && Y.is_cuda(), "X, Y must be CUDA tensors");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && Y.dtype() == torch::kFloat32,
                "X, Y must be float32");
    TORCH_CHECK(X.dim() == 2 && Y.dim() == 2, "X, Y must be 2D");
    TORCH_CHECK(X.size(1) == Y.size(1), "X, Y must share feature dim D");
    TORCH_CHECK(X.size(1) % 4 == 0, "D must be multiple of 4 for float4 path");

    X = X.contiguous();
    Y = Y.contiguous();
    int N = X.size(0);
    int M = Y.size(0);
    int D = X.size(1);

    auto K = torch::empty({N, M}, X.options());

    dim3 threads(32, 8);   // 256 线程/block
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    rbf_forward_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), K.data_ptr<float>(),
        N, M, D, (float)gamma);

    return K;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbf_forward", &rbf_forward, "RBF kernel matrix forward (CUDA, minimal L2)");
}
