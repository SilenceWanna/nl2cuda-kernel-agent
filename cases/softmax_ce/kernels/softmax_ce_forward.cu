#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cfloat>

namespace {

constexpr int THREADS = 256;

template <bool STORE_PROBS>
__global__ void softmax_ce_forward_rows_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ labels,
    float* __restrict__ probs,
    float* __restrict__ loss,
    int bsz,
    int classes) {
    __shared__ float shared[THREADS];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= bsz) {
        return;
    }

    const float* row_logits = logits + static_cast<long long>(row) * classes;
    float local_max = -FLT_MAX;
    for (int c = tid; c < classes; c += blockDim.x) {
        float v = row_logits[c];
        local_max = fmaxf(local_max, v);
    }

    shared[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    float local_sum = 0.0f;
    float* row_probs = probs;
    if (STORE_PROBS) {
        row_probs = probs + static_cast<long long>(row) * classes;
    }
    for (int c = tid; c < classes; c += blockDim.x) {
        float e = expf(row_logits[c] - row_max);
        local_sum += e;
        if (STORE_PROBS) {
            row_probs[c] = e;
        }
    }

    shared[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = shared[0];

    if (STORE_PROBS) {
        float inv_sum = 1.0f / row_sum;
        for (int c = tid; c < classes; c += blockDim.x) {
            row_probs[c] *= inv_sum;
        }
    }

    if (tid == 0) {
        int64_t label = labels[row];
        float row_loss = row_max + logf(row_sum) - row_logits[label];
        atomicAdd(loss, row_loss / static_cast<float>(bsz));
    }
}

void check_softmax_ce_inputs(torch::Tensor logits, torch::Tensor labels) {
    TORCH_CHECK(logits.is_cuda() && labels.is_cuda(),
                "softmax_ce inputs must be CUDA tensors");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
                "softmax_ce logits must be float32");
    TORCH_CHECK(labels.scalar_type() == torch::kInt64,
                "softmax_ce labels must be int64");
    TORCH_CHECK(logits.dim() == 2, "softmax_ce logits must be 2D");
    TORCH_CHECK(labels.dim() == 1, "softmax_ce labels must be 1D");
    TORCH_CHECK(logits.size(0) == labels.size(0),
                "softmax_ce batch size mismatch");
    TORCH_CHECK(logits.is_contiguous() && labels.is_contiguous(),
                "softmax_ce inputs must be contiguous");
}

}  // namespace

torch::Tensor softmax_ce_forward_only(torch::Tensor logits, torch::Tensor labels) {
    check_softmax_ce_inputs(logits, labels);

    int bsz = static_cast<int>(logits.size(0));
    int classes = static_cast<int>(logits.size(1));
    auto loss = torch::zeros({}, logits.options());

    softmax_ce_forward_rows_kernel<false>
        <<<bsz, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            logits.data_ptr<float>(),
            labels.data_ptr<int64_t>(),
            nullptr,
            loss.data_ptr<float>(),
            bsz,
            classes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return loss;
}

std::vector<torch::Tensor> softmax_ce_forward(
    torch::Tensor logits,
    torch::Tensor labels) {
    check_softmax_ce_inputs(logits, labels);

    int bsz = static_cast<int>(logits.size(0));
    int classes = static_cast<int>(logits.size(1));
    auto loss = torch::zeros({}, logits.options());
    auto probs = torch::empty_like(logits);

    softmax_ce_forward_rows_kernel<true>
        <<<bsz, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            logits.data_ptr<float>(),
            labels.data_ptr<int64_t>(),
            probs.data_ptr<float>(),
            loss.data_ptr<float>(),
            bsz,
            classes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {loss, probs};
}
