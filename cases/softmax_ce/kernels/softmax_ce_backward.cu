// Softmax 交叉熵 —— 反向 kernel（block-per-row）。
//   上游标量梯度 gout = dL/dloss。dlogits[b,c] = gout * (softmax[b,c] - onehot[b,c]) / B
//
// 一个 block 处理一行 b：规约求 max + sum(exp) 得 softmax；
// 每类别 c 写 dlogits[b,c] = gout/B * (softmax[b,c] - (c==labels[b]?1:0))。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS 256

__global__ void softmax_ce_backward_kernel(
        const float* __restrict__ logits,   // [B, C]
        const long*  __restrict__ labels,   // [B]
        const float* __restrict__ gout,     // [1] 上游标量梯度
        float* __restrict__ dlogits,        // [B, C]
        int B, int C) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float* row = logits + (size_t)b * C;
    float* drow = dlogits + (size_t)b * C;

    __shared__ float sred[THREADS];
    int t = threadIdx.x;

    // row max
    float lmax = -3.4e38f;
    for (int c = t; c < C; c += blockDim.x) lmax = fmaxf(lmax, row[c]);
    sred[t] = lmax; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sred[t] = fmaxf(sred[t], sred[t + s]);
        __syncthreads();
    }
    float rowmax = sred[0]; __syncthreads();

    // sum(exp)
    float lsum = 0.0f;
    for (int c = t; c < C; c += blockDim.x) lsum += expf(row[c] - rowmax);
    sred[t] = lsum; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sred[t] += sred[t + s];
        __syncthreads();
    }
    float rowsum = sred[0]; __syncthreads();

    float scale = gout[0] / (float)B;
    long lbl = labels[b];
    float inv = 1.0f / rowsum;
    for (int c = t; c < C; c += blockDim.x) {
        float sm = expf(row[c] - rowmax) * inv;      // softmax[b,c]
        float onehot = (c == lbl) ? 1.0f : 0.0f;
        drow[c] = scale * (sm - onehot);
    }
}

torch::Tensor softmax_ce_backward(torch::Tensor logits, torch::Tensor labels,
                                  torch::Tensor gout) {
    TORCH_CHECK(logits.is_cuda() && labels.is_cuda() && gout.is_cuda(), "inputs must be CUDA");
    logits = logits.contiguous();
    labels = labels.contiguous();
    gout = gout.contiguous();
    int B = logits.size(0), C = logits.size(1);

    auto dlogits = torch::empty_like(logits);
    softmax_ce_backward_kernel<<<B, THREADS>>>(
        logits.data_ptr<float>(), labels.data_ptr<long>(),
        gout.data_ptr<float>(), dlogits.data_ptr<float>(), B, C);
    return dlogits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_ce_backward", &softmax_ce_backward, "Softmax CE backward (CUDA)");
}
