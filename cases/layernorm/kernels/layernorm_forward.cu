// LayerNorm 前向 kernel（朴素正确版，不追求速度）。
//   输入 X:[B,D], gamma:[D], beta:[D]；在最后一维 D 归一化。
//   mean = mean_d X[b,d];  var = mean_d (X[b,d]-mean)^2
//   Y[b,d] = (X[b,d]-mean)/sqrt(var+eps) * gamma[d] + beta[d]
//
// 策略：一个 block 处理一行 b，blockDim=256 个线程协作规约求 mean/var（shared memory），
// 再各线程写回该行的 Y。朴素但正确；优化留到后续。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void layernorm_forward_kernel(
        const float* __restrict__ X,      // [B, D]
        const float* __restrict__ gamma,  // [D]
        const float* __restrict__ beta,   // [D]
        float* __restrict__ Y,            // [B, D]
        int B, int D, float eps) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float* xrow = X + (size_t)b * D;
    float* yrow = Y + (size_t)b * D;

    extern __shared__ float sdata[];  // blockDim.x 个元素，用于规约

    // --- 求和 -> mean ---
    float local = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) local += xrow[d];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / D;
    __syncthreads();

    // --- 求 (x-mean)^2 之和 -> var ---
    float localv = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float diff = xrow[d] - mean;
        localv += diff * diff;
    }
    sdata[threadIdx.x] = localv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float var = sdata[0] / D;
    float rstd = rsqrtf(var + eps);

    // --- 写回 ---
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xhat = (xrow[d] - mean) * rstd;
        yrow[d] = xhat * gamma[d] + beta[d];
    }
}

torch::Tensor layernorm_forward(torch::Tensor X, torch::Tensor gamma,
                                torch::Tensor beta, double eps) {
    TORCH_CHECK(X.is_cuda() && gamma.is_cuda() && beta.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(X.dim() == 2, "X must be 2D [B,D]");
    X = X.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();
    int B = X.size(0);
    int D = X.size(1);

    auto Y = torch::empty_like(X);
    const int threads = 256;
    layernorm_forward_kernel<<<B, threads, threads * sizeof(float)>>>(
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        Y.data_ptr<float>(), B, D, (float)eps);
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA, naive)");
}
