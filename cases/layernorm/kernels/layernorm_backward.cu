// LayerNorm 反向 kernel（朴素正确版）。
// 记 xhat=(x-mean)/std, std=sqrt(var+eps), 上游梯度 G=dL/dY。
//   dgamma[d] = Σ_b G[b,d]*xhat[b,d]
//   dbeta[d]  = Σ_b G[b,d]
//   g1[b,d]=G[b,d]*gamma[d];  m1_b=mean_d g1[b,:];  m2_b=mean_d (g1[b,:]*xhat[b,:])
//   dX[b,d] = (1/std_b) * ( g1[b,d] - m1_b - xhat[b,d]*m2_b )
//
// 策略：
//   dX kernel：一个 block 处理一行 b，block 内重算 mean/std，再规约 m1/m2，写 dX 行。
//   dgamma/dbeta kernel：一个线程负责一列 d，沿 B 归约（朴素）。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 一个 block 处理一行 b，求 dX[b,:]
__global__ void layernorm_backward_dX_kernel(
        const float* __restrict__ X,      // [B,D]
        const float* __restrict__ G,      // [B,D] 上游
        const float* __restrict__ gamma,  // [D]
        float* __restrict__ dX,           // [B,D]
        int B, int D, float eps) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float* xrow = X + (size_t)b * D;
    const float* grow = G + (size_t)b * D;
    float* dxrow = dX + (size_t)b * D;

    extern __shared__ float sdata[];

    // mean
    float local = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) local += xrow[d];
    sdata[threadIdx.x] = local; __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1) { if (threadIdx.x<s) sdata[threadIdx.x]+=sdata[threadIdx.x+s]; __syncthreads(); }
    float mean = sdata[0]/D; __syncthreads();

    // var
    float localv = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) { float df=xrow[d]-mean; localv+=df*df; }
    sdata[threadIdx.x]=localv; __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1){ if(threadIdx.x<s) sdata[threadIdx.x]+=sdata[threadIdx.x+s]; __syncthreads(); }
    float var = sdata[0]/D;
    float rstd = rsqrtf(var+eps); __syncthreads();

    // m1 = mean_d g1 ;  g1 = G*gamma
    float lm1 = 0.0f;
    for (int d=threadIdx.x; d<D; d+=blockDim.x) lm1 += grow[d]*gamma[d];
    sdata[threadIdx.x]=lm1; __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1){ if(threadIdx.x<s) sdata[threadIdx.x]+=sdata[threadIdx.x+s]; __syncthreads(); }
    float m1 = sdata[0]/D; __syncthreads();

    // m2 = mean_d (g1 * xhat)
    float lm2 = 0.0f;
    for (int d=threadIdx.x; d<D; d+=blockDim.x) {
        float xhat = (xrow[d]-mean)*rstd;
        lm2 += (grow[d]*gamma[d]) * xhat;
    }
    sdata[threadIdx.x]=lm2; __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1){ if(threadIdx.x<s) sdata[threadIdx.x]+=sdata[threadIdx.x+s]; __syncthreads(); }
    float m2 = sdata[0]/D; __syncthreads();

    // dX[b,d] = rstd * ( g1 - m1 - xhat*m2 )
    for (int d=threadIdx.x; d<D; d+=blockDim.x) {
        float xhat = (xrow[d]-mean)*rstd;
        float g1 = grow[d]*gamma[d];
        dxrow[d] = rstd * (g1 - m1 - xhat*m2);
    }
}

// 一个线程负责一列 d，沿 B 归约求 dgamma[d]、dbeta[d]
__global__ void layernorm_backward_dparam_kernel(
        const float* __restrict__ X,      // [B,D]
        const float* __restrict__ G,      // [B,D]
        float* __restrict__ dgamma,       // [D]
        float* __restrict__ dbeta,        // [D]
        int B, int D, float eps) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    float dg = 0.0f, db = 0.0f;
    for (int b = 0; b < B; ++b) {
        const float* xrow = X + (size_t)b * D;
        // 重算该行 mean/std
        float mean = 0.0f;
        for (int k = 0; k < D; ++k) mean += xrow[k];
        mean /= D;
        float var = 0.0f;
        for (int k = 0; k < D; ++k) { float df = xrow[k]-mean; var += df*df; }
        var /= D;
        float rstd = rsqrtf(var + eps);
        float xhat = (xrow[d] - mean) * rstd;
        float g = G[(size_t)b * D + d];
        dg += g * xhat;
        db += g;
    }
    dgamma[d] = dg;
    dbeta[d] = db;
}

std::vector<torch::Tensor> layernorm_backward(
        torch::Tensor X, torch::Tensor G, torch::Tensor gamma, double eps) {
    TORCH_CHECK(X.is_cuda() && G.is_cuda() && gamma.is_cuda(), "inputs must be CUDA");
    X = X.contiguous(); G = G.contiguous(); gamma = gamma.contiguous();
    int B = X.size(0), D = X.size(1);

    auto dX = torch::empty_like(X);
    auto dgamma = torch::empty({D}, X.options());
    auto dbeta = torch::empty({D}, X.options());

    const int threads = 256;
    layernorm_backward_dX_kernel<<<B, threads, threads*sizeof(float)>>>(
        X.data_ptr<float>(), G.data_ptr<float>(), gamma.data_ptr<float>(),
        dX.data_ptr<float>(), B, D, (float)eps);
    layernorm_backward_dparam_kernel<<<(D+threads-1)/threads, threads>>>(
        X.data_ptr<float>(), G.data_ptr<float>(),
        dgamma.data_ptr<float>(), dbeta.data_ptr<float>(), B, D, (float)eps);

    return {dX, dgamma, dbeta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_backward", &layernorm_backward, "LayerNorm backward (CUDA, naive)");
}
