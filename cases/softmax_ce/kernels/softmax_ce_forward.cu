#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        shared[wid] = v;
    }
    __syncthreads();

    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
    }
    if (threadIdx.x == 0) {
        shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

__inline__ __device__ float block_reduce_max(float v) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    v = warp_reduce_max(v);
    if (lane == 0) {
        shared[wid] = v;
    }
    __syncthreads();

    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        v = warp_reduce_max(v);
    }
    if (threadIdx.x == 0) {
        shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

__global__ void softmax_ce_forward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ labels,
    float* __restrict__ loss_sum,
    float* __restrict__ logsumexp,
    int B,
    int C) {
    const int b = blockIdx.x;
    if (b >= B) {
        return;
    }

    const int row = b * C;

    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        local_max = fmaxf(local_max, logits[row + c]);
    }
    const float row_max = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        local_sum += expf(logits[row + c] - row_max);
    }
    const float exp_sum = block_reduce_sum(local_sum);
    const float lse = row_max + logf(exp_sum);

    if (threadIdx.x == 0) {
        const int64_t label = labels[b];
        const float row_loss = lse - logits[row + static_cast<int>(label)];
        logsumexp[b] = lse;
        atomicAdd(loss_sum, row_loss / static_cast<float>(B));
    }
}

void check_forward_inputs(const torch::Tensor& logits, const torch::Tensor& labels) {
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
    TORCH_CHECK(labels.is_cuda(), "labels must be CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(labels.scalar_type() == torch::kInt64, "labels must be int64");
    TORCH_CHECK(logits.dim() == 2, "logits must have shape [B, C]");
    TORCH_CHECK(labels.dim() == 1, "labels must have shape [B]");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(labels.is_contiguous(), "labels must be contiguous");
    TORCH_CHECK(labels.size(0) == logits.size(0), "labels length must equal B");
}

}  // namespace

std::vector<torch::Tensor> softmax_ce_forward(torch::Tensor logits, torch::Tensor labels) {
    check_forward_inputs(logits, labels);

    const int B = static_cast<int>(logits.size(0));
    const int C = static_cast<int>(logits.size(1));

    auto scalar_opts = logits.options().dtype(torch::kFloat32);
    auto loss_sum = torch::zeros({}, scalar_opts);
    auto logsumexp = torch::empty({B}, scalar_opts);

    softmax_ce_forward_kernel<<<B, kBlockSize>>>(
        logits.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        loss_sum.data_ptr<float>(),
        logsumexp.data_ptr<float>(),
        B,
        C);

    return {loss_sum, logsumexp};
}
