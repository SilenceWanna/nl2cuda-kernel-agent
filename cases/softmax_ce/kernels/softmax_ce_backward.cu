#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

std::vector<torch::Tensor> softmax_ce_forward(torch::Tensor logits, torch::Tensor labels);

namespace {

constexpr int kBlockSize = 256;

__global__ void softmax_ce_backward_kernel(
    const float* __restrict__ grad_loss,
    const float* __restrict__ logits,
    const int64_t* __restrict__ labels,
    const float* __restrict__ logsumexp,
    float* __restrict__ grad_logits,
    int B,
    int C) {
    const int b = blockIdx.x;

    if (b >= B) {
        return;
    }

    const int row = b * C;
    const int64_t label = labels[b];
    const float scale = grad_loss[0] / static_cast<float>(B);
    const float lse = logsumexp[b];

    for (int col = threadIdx.x; col < C; col += blockDim.x) {
        float grad = expf(logits[row + col] - lse);
        if (col == static_cast<int>(label)) {
            grad -= 1.0f;
        }
        grad_logits[row + col] = grad * scale;
    }
}

void check_backward_inputs(
    const torch::Tensor& grad_loss,
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    const torch::Tensor& logsumexp) {
    TORCH_CHECK(grad_loss.is_cuda(), "grad_loss must be CUDA");
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
    TORCH_CHECK(labels.is_cuda(), "labels must be CUDA");
    TORCH_CHECK(logsumexp.is_cuda(), "logsumexp must be CUDA");
    TORCH_CHECK(grad_loss.scalar_type() == torch::kFloat32, "grad_loss must be float32");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(labels.scalar_type() == torch::kInt64, "labels must be int64");
    TORCH_CHECK(logsumexp.scalar_type() == torch::kFloat32, "logsumexp must be float32");
    TORCH_CHECK(logits.dim() == 2, "logits must have shape [B, C]");
    TORCH_CHECK(labels.dim() == 1, "labels must have shape [B]");
    TORCH_CHECK(logsumexp.dim() == 1, "logsumexp must have shape [B]");
    TORCH_CHECK(grad_loss.numel() == 1, "grad_loss must be scalar");
    TORCH_CHECK(grad_loss.is_contiguous(), "grad_loss must be contiguous");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(labels.is_contiguous(), "labels must be contiguous");
    TORCH_CHECK(logsumexp.is_contiguous(), "logsumexp must be contiguous");
    TORCH_CHECK(labels.size(0) == logits.size(0), "labels length must equal B");
    TORCH_CHECK(logsumexp.size(0) == logits.size(0), "logsumexp length must equal B");
}

}  // namespace

torch::Tensor softmax_ce_backward(
    torch::Tensor grad_loss,
    torch::Tensor logits,
    torch::Tensor labels,
    torch::Tensor logsumexp) {
    check_backward_inputs(grad_loss, logits, labels, logsumexp);

    const int B = static_cast<int>(logits.size(0));
    const int C = static_cast<int>(logits.size(1));

    auto grad_logits = torch::empty_like(logits);

    softmax_ce_backward_kernel<<<B, kBlockSize>>>(
        grad_loss.data_ptr<float>(),
        logits.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        logsumexp.data_ptr<float>(),
        grad_logits.data_ptr<float>(),
        B,
        C);

    return grad_logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_ce_forward", &softmax_ce_forward, "Softmax cross-entropy forward (CUDA)");
    m.def("softmax_ce_backward", &softmax_ce_backward, "Softmax cross-entropy backward (CUDA)");
}
