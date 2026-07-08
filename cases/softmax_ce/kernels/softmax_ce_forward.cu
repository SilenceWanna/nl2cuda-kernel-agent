// Softmax 交叉熵 —— 前向 kernel（block-per-row，数值稳定 logsumexp）。
//   logits:[B,C], labels:[B] -> 标量 loss = mean_b( logsumexp(logits[b,:]) - logits[b,labels[b]] )
//
// 一个 block 处理一行 b：blockDim 个线程 grid-stride 遍历 C 个类别，
// block 内规约求 max（数值稳定）与 sum(exp(logit-max))，得 logsumexp；
// loss_b = logsumexp - logits[b, labels[b]]；atomicAdd 累加到全局 loss（host 端 /B）。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS 256

__global__ void softmax_ce_forward_kernel(
        const float* __restrict__ logits,   // [B, C]
        const long*  __restrict__ labels,   // [B]
        float* __restrict__ loss_accum,     // [1] 全局累加器
        int B, int C) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float* row = logits + (size_t)b * C;

    __shared__ float sred[THREADS];
    int t = threadIdx.x;

    // --- 求行 max ---
    float lmax = -3.4e38f;
    for (int c = t; c < C; c += blockDim.x) lmax = fmaxf(lmax, row[c]);
    sred[t] = lmax; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sred[t] = fmaxf(sred[t], sred[t + s]);
        __syncthreads();
    }
    float rowmax = sred[0];
    __syncthreads();

    // --- 求 sum(exp(x-max)) ---
    float lsum = 0.0f;
    for (int c = t; c < C; c += blockDim.x) lsum += expf(row[c] - rowmax);
    sred[t] = lsum; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sred[t] += sred[t + s];
        __syncthreads();
    }
    if (t == 0) {
        float logsumexp = rowmax + logf(sred[0]);
        float loss_b = logsumexp - row[labels[b]];
        atomicAdd(loss_accum, loss_b);
    }
}

torch::Tensor softmax_ce_forward(torch::Tensor logits, torch::Tensor labels) {
    TORCH_CHECK(logits.is_cuda() && labels.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(labels.dtype() == torch::kLong, "labels must be int64");
    logits = logits.contiguous();
    labels = labels.contiguous();
    int B = logits.size(0), C = logits.size(1);

    auto loss = torch::zeros({}, logits.options());
    softmax_ce_forward_kernel<<<B, THREADS>>>(
        logits.data_ptr<float>(), labels.data_ptr<long>(),
        loss.data_ptr<float>(), B, C);
    return loss / B;   // 对 batch 取平均
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_ce_forward", &softmax_ce_forward, "Softmax CE forward (CUDA)");
}
