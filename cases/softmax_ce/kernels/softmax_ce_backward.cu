#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;

__global__ void softmax_ce_backward_kernel(
        const float* __restrict__ logits,
        const int64_t* __restrict__ labels,
        const float* __restrict__ grad_loss,
        const float* __restrict__ row_max,
        const float* __restrict__ inv_sum,
        float* __restrict__ dlogits,
        int B,
        int C) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B) return;

    const float* row = logits + (size_t)b * C;
    float* drow = dlogits + (size_t)b * C;
    int64_t y = labels[b];
    float m = row_max[b];
    float inv = inv_sum[b];
    float scale = grad_loss[0] / static_cast<float>(B);

    for (int c = tid; c < C; c += THREADS) {
        float p = expf(row[c] - m) * inv;
        float target = (c == y) ? 1.0f : 0.0f;
        drow[c] = (p - target) * scale;
    }
}

}  // namespace

torch::Tensor softmax_ce_backward(
        torch::Tensor logits,
        torch::Tensor labels,
        torch::Tensor grad_loss,
        torch::Tensor row_max,
        torch::Tensor inv_sum) {
    TORCH_CHECK(logits.is_cuda() && labels.is_cuda() && grad_loss.is_cuda()
                && row_max.is_cuda() && inv_sum.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(logits.dtype() == torch::kFloat32 && grad_loss.dtype() == torch::kFloat32
                && row_max.dtype() == torch::kFloat32 && inv_sum.dtype() == torch::kFloat32,
                "floating tensors must be float32");
    TORCH_CHECK(labels.dtype() == torch::kInt64, "labels must be int64");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
    TORCH_CHECK(labels.dim() == 1, "labels must be 1D");
    TORCH_CHECK(grad_loss.numel() == 1, "grad_loss must be scalar");

    logits = logits.contiguous();
    labels = labels.contiguous();
    grad_loss = grad_loss.contiguous();
    row_max = row_max.contiguous();
    inv_sum = inv_sum.contiguous();

    int B = static_cast<int>(logits.size(0));
    int C = static_cast<int>(logits.size(1));

    TORCH_CHECK(labels.size(0) == B, "labels length must match batch size");
    TORCH_CHECK(row_max.numel() == B && inv_sum.numel() == B, "cached tensors must match B");

    auto dlogits = torch::empty_like(logits);
    softmax_ce_backward_kernel<<<B, THREADS>>>(
        logits.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        grad_loss.data_ptr<float>(),
        row_max.data_ptr<float>(),
        inv_sum.data_ptr<float>(),
        dlogits.data_ptr<float>(),
        B,
        C);

    return dlogits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_ce_backward", &softmax_ce_backward,
          "Mean softmax cross-entropy backward (CUDA)");
}
