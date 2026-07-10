#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int THREADS = 256;

__global__ void softmax_ce_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ probs,
    const int64_t* __restrict__ labels,
    float* __restrict__ grad_logits,
    int bsz,
    int classes,
    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int row = idx / classes;
    int col = idx - row * classes;
    float grad = probs[idx];
    if (col == labels[row]) {
        grad -= 1.0f;
    }
    grad_logits[idx] = grad * grad_out[0] / static_cast<float>(bsz);
}

}  // namespace

torch::Tensor softmax_ce_forward_only(torch::Tensor logits, torch::Tensor labels);
std::vector<torch::Tensor> softmax_ce_forward(
    torch::Tensor logits,
    torch::Tensor labels);

torch::Tensor softmax_ce_backward(
    torch::Tensor grad_out,
    torch::Tensor probs,
    torch::Tensor labels) {
    TORCH_CHECK(grad_out.is_cuda() && probs.is_cuda() && labels.is_cuda(),
                "softmax_ce backward tensors must be CUDA tensors");
    TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 &&
                probs.scalar_type() == torch::kFloat32,
                "softmax_ce backward only supports float32");
    TORCH_CHECK(labels.scalar_type() == torch::kInt64,
                "softmax_ce backward labels must be int64");
    TORCH_CHECK(grad_out.numel() == 1,
                "softmax_ce backward expects scalar grad_out");
    TORCH_CHECK(probs.dim() == 2 && labels.dim() == 1,
                "softmax_ce backward shape mismatch");
    TORCH_CHECK(probs.size(0) == labels.size(0),
                "softmax_ce backward batch size mismatch");
    TORCH_CHECK(grad_out.is_contiguous() && probs.is_contiguous() && labels.is_contiguous(),
                "softmax_ce backward tensors must be contiguous");

    int bsz = static_cast<int>(probs.size(0));
    int classes = static_cast<int>(probs.size(1));
    int total = bsz * classes;
    auto grad_logits = torch::empty_like(probs);

    int blocks = (total + THREADS - 1) / THREADS;
    softmax_ce_backward_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out.data_ptr<float>(),
        probs.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        grad_logits.data_ptr<float>(),
        bsz,
        classes,
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_ce_forward_only", &softmax_ce_forward_only,
          "Softmax cross-entropy forward loss only (CUDA)");
    m.def("softmax_ce_forward", &softmax_ce_forward,
          "Softmax cross-entropy forward with probabilities (CUDA)");
    m.def("softmax_ce_backward", &softmax_ce_backward,
          "Softmax cross-entropy backward (CUDA)");
}
