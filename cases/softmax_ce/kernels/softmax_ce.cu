#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cfloat>
#include <cstdint>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INT64(x) TORCH_CHECK(x.scalar_type() == torch::kInt64, #x " must be int64")
#define CHECK_INPUT_FLOAT(x) \
    CHECK_CUDA(x);           \
    CHECK_CONTIGUOUS(x);     \
    CHECK_FLOAT(x)
#define CHECK_INPUT_INT64(x) \
    CHECK_CUDA(x);           \
    CHECK_CONTIGUOUS(x);     \
    CHECK_INT64(x)

namespace {

__global__ void softmax_ce_forward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ labels,
    float* __restrict__ probs,
    float* __restrict__ row_losses,
    int B,
    int C) {
    extern __shared__ float shared[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= B) {
        return;
    }

    const float* row_logits = logits + static_cast<int64_t>(row) * C;
    float* row_probs = probs + static_cast<int64_t>(row) * C;

    float local_max = -FLT_MAX;
    for (int c = tid; c < C; c += blockDim.x) {
        const float v = row_logits[c];
        local_max = v > local_max ? v : local_max;
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = shared[tid + stride];
            shared[tid] = other > shared[tid] ? other : shared[tid];
        }
        __syncthreads();
    }

    const float row_max = shared[0];

    float local_sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        const float e = expf(row_logits[c] - row_max);
        row_probs[c] = e;
        local_sum += e;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    const float sum_exp = shared[0];
    const float inv_sum_exp = 1.0f / sum_exp;

    for (int c = tid; c < C; c += blockDim.x) {
        row_probs[c] *= inv_sum_exp;
    }

    if (tid == 0) {
        const int64_t label = labels[row];
        row_losses[row] = row_max + logf(sum_exp) - row_logits[label];
    }
}

__global__ void reduce_loss_kernel(
    const float* __restrict__ row_losses,
    float* __restrict__ loss,
    int B) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int i = tid; i < B; i += blockDim.x) {
        local_sum += row_losses[i];
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        loss[0] = shared[0] / static_cast<float>(B);
    }
}

__global__ void softmax_ce_backward_kernel(
    const float* __restrict__ probs,
    const int64_t* __restrict__ labels,
    const float* __restrict__ grad_loss,
    float* __restrict__ grad_logits,
    int64_t total,
    int B,
    int C) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    const int row = static_cast<int>(idx / C);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * C);
    const float scale = grad_loss[0] / static_cast<float>(B);
    const float target = (col == labels[row]) ? 1.0f : 0.0f;

    grad_logits[idx] = scale * (probs[idx] - target);
}

}  // namespace

std::vector<torch::Tensor> softmax_ce_forward(torch::Tensor logits, torch::Tensor labels) {
    CHECK_INPUT_FLOAT(logits);
    CHECK_INPUT_INT64(labels);

    TORCH_CHECK(logits.dim() == 2, "logits must have shape [B, C]");
    TORCH_CHECK(labels.dim() == 1, "labels must have shape [B]");
    TORCH_CHECK(logits.size(0) == labels.size(0), "labels length must match logits batch size");

    const int B = static_cast<int>(logits.size(0));
    const int C = static_cast<int>(logits.size(1));

    auto probs = torch::empty_like(logits);
    auto row_losses = torch::empty({B}, logits.options());
    auto loss = torch::empty({}, logits.options());

    const int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    softmax_ce_forward_kernel<<<B, threads, shared_bytes, stream>>>(
        logits.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        probs.data_ptr<float>(),
        row_losses.data_ptr<float>(),
        B,
        C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    reduce_loss_kernel<<<1, threads, shared_bytes, stream>>>(
        row_losses.data_ptr<float>(),
        loss.data_ptr<float>(),
        B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {loss, probs};
}

torch::Tensor softmax_ce_backward(
    torch::Tensor probs,
    torch::Tensor labels,
    torch::Tensor grad_loss) {
    CHECK_INPUT_FLOAT(probs);
    CHECK_INPUT_INT64(labels);
    CHECK_INPUT_FLOAT(grad_loss);

    TORCH_CHECK(probs.dim() == 2, "probs must have shape [B, C]");
    TORCH_CHECK(labels.dim() == 1, "labels must have shape [B]");
    TORCH_CHECK(grad_loss.numel() == 1, "grad_loss must be scalar");
    TORCH_CHECK(probs.size(0) == labels.size(0), "labels length must match probs batch size");

    const int B = static_cast<int>(probs.size(0));
    const int C = static_cast<int>(probs.size(1));
    const int64_t total = probs.numel();

    auto grad_logits = torch::empty_like(probs);

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    softmax_ce_backward_kernel<<<blocks, threads, 0, stream>>>(
        probs.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        grad_loss.data_ptr<float>(),
        grad_logits.data_ptr<float>(),
        total,
        B,
        C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad_logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_ce_forward, "Softmax cross-entropy forward");
    m.def("backward", &softmax_ce_backward, "Softmax cross-entropy backward");
}
