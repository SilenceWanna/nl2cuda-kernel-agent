// Softmax cross-entropy backward kernel, fp32.
// Given upstream scalar grad_out, compute:
//   dlogits[b,c] = grad_out * (softmax(logits)[b,c] - onehot(labels[b])[c]) / B
//
// The backward recomputes the per-row max and exp-sum for numerical stability rather
// than relying on any high-level torch op or reduced-precision approximation.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;

__global__ void softmax_ce_backward_kernel(
        const float* __restrict__ logits,
        const int64_t* __restrict__ labels,
        const float* __restrict__ grad_out,
        float* __restrict__ dlogits,
        int B,
        int C) {
    __shared__ float smem[THREADS];

    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B) return;

    const float* row = logits + (size_t)b * C;
    float* drow = dlogits + (size_t)b * C;

    float local_max = -CUDART_INF_F;
    for (int c = tid; c < C; c += blockDim.x) {
        local_max = fmaxf(local_max, row[c]);
    }
    smem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = smem[0];

    float local_sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        local_sum += expf(row[c] - row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / smem[0];
    float scale = grad_out[0] / (float)B;
    int64_t label = labels[b];

    for (int c = tid; c < C; c += blockDim.x) {
        float p = expf(row[c] - row_max) * inv_sum;
        float target = (c == label) ? 1.0f : 0.0f;
        drow[c] = scale * (p - target);
    }
}

}  // namespace

torch::Tensor softmax_ce_backward(
        torch::Tensor logits,
        torch::Tensor labels,
        torch::Tensor grad_out) {
    TORCH_CHECK(logits.is_cuda() && labels.is_cuda() && grad_out.is_cuda(),
                "logits, labels, and grad_out must be CUDA tensors");
    TORCH_CHECK(logits.dtype() == torch::kFloat32 && grad_out.dtype() == torch::kFloat32,
                "logits and grad_out must be float32");
    TORCH_CHECK(labels.dtype() == torch::kInt64, "labels must be int64");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [B,C]");
    TORCH_CHECK(labels.dim() == 1, "labels must be 1D [B]");
    TORCH_CHECK(labels.size(0) == logits.size(0), "labels length must equal B");
    TORCH_CHECK(grad_out.numel() == 1, "grad_out must be a scalar");

    logits = logits.contiguous();
    labels = labels.contiguous();
    grad_out = grad_out.contiguous();

    int B = (int)logits.size(0);
    int C = (int)logits.size(1);
    auto dlogits = torch::empty_like(logits);

    softmax_ce_backward_kernel<<<B, THREADS>>>(
        logits.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        grad_out.data_ptr<float>(),
        dlogits.data_ptr<float>(),
        B,
        C);

    return dlogits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_ce_backward", &softmax_ce_backward,
          "Softmax cross-entropy backward (CUDA)");
}
