// Softmax cross-entropy forward kernel, fp32, numerically stable logsumexp.
// logits:[B,C] float32, labels:[B] int64 -> scalar mean loss.
//
// One CUDA block handles one row. Threads reduce row max and exp-sum in shared memory,
// then one thread atomically accumulates the row loss into a scalar initialized to 0.
// No fast-math flags are used by framework.loader; expf/logf are the regular fp32 funcs.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;

__global__ void softmax_ce_forward_kernel(
        const float* __restrict__ logits,
        const int64_t* __restrict__ labels,
        float* __restrict__ loss,
        int B,
        int C) {
    __shared__ float smem[THREADS];

    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B) return;

    const float* row = logits + (size_t)b * C;

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

    if (tid == 0) {
        int64_t label = labels[b];
        float chosen = row[label];
        float row_loss = logf(smem[0]) + row_max - chosen;
        atomicAdd(loss, row_loss / (float)B);
    }
}

}  // namespace

torch::Tensor softmax_ce_forward(torch::Tensor logits, torch::Tensor labels) {
    TORCH_CHECK(logits.is_cuda() && labels.is_cuda(), "logits and labels must be CUDA tensors");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(labels.dtype() == torch::kInt64, "labels must be int64");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [B,C]");
    TORCH_CHECK(labels.dim() == 1, "labels must be 1D [B]");
    TORCH_CHECK(labels.size(0) == logits.size(0), "labels length must equal B");

    logits = logits.contiguous();
    labels = labels.contiguous();

    int B = (int)logits.size(0);
    int C = (int)logits.size(1);
    auto loss = torch::zeros({}, logits.options());

    softmax_ce_forward_kernel<<<B, THREADS>>>(
        logits.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        loss.data_ptr<float>(),
        B,
        C);

    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_ce_forward", &softmax_ce_forward,
          "Softmax cross-entropy forward (CUDA)");
}
