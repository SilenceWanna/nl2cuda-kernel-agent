#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

namespace {

constexpr int THREADS = 256;

__device__ float block_reduce_max(float v) {
    __shared__ float buf[THREADS];
    int tid = threadIdx.x;
    buf[tid] = v;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = buf[tid + stride];
            buf[tid] = other > buf[tid] ? other : buf[tid];
        }
        __syncthreads();
    }
    return buf[0];
}

__device__ float block_reduce_sum(float v) {
    __shared__ float buf[THREADS];
    int tid = threadIdx.x;
    buf[tid] = v;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            buf[tid] += buf[tid + stride];
        }
        __syncthreads();
    }
    return buf[0];
}

__global__ void softmax_ce_forward_kernel(
        const float* __restrict__ logits,
        const int64_t* __restrict__ labels,
        float* __restrict__ loss,
        float* __restrict__ row_max,
        float* __restrict__ inv_sum,
        int B,
        int C) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B) return;

    const float* row = logits + (size_t)b * C;

    float local_max = -FLT_MAX;
    for (int c = tid; c < C; c += THREADS) {
        float v = row[c];
        local_max = v > local_max ? v : local_max;
    }
    float m = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int c = tid; c < C; c += THREADS) {
        local_sum += expf(row[c] - m);
    }
    float s = block_reduce_sum(local_sum);

    if (tid == 0) {
        int64_t y = labels[b];
        float inv = 1.0f / s;
        row_max[b] = m;
        inv_sum[b] = inv;
        float row_loss = logf(s) + m - row[y];
        atomicAdd(loss, row_loss / static_cast<float>(B));
    }
}

}  // namespace

std::vector<torch::Tensor> softmax_ce_forward(torch::Tensor logits, torch::Tensor labels) {
    TORCH_CHECK(logits.is_cuda() && labels.is_cuda(), "logits and labels must be CUDA tensors");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(labels.dtype() == torch::kInt64, "labels must be int64");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
    TORCH_CHECK(labels.dim() == 1, "labels must be 1D");
    TORCH_CHECK(logits.size(0) == labels.size(0), "labels length must match batch size");

    logits = logits.contiguous();
    labels = labels.contiguous();

    int B = static_cast<int>(logits.size(0));
    int C = static_cast<int>(logits.size(1));

    auto loss = torch::zeros({}, logits.options());
    auto row_max = torch::empty({B}, logits.options());
    auto inv_sum = torch::empty({B}, logits.options());

    softmax_ce_forward_kernel<<<B, THREADS>>>(
        logits.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        loss.data_ptr<float>(),
        row_max.data_ptr<float>(),
        inv_sum.data_ptr<float>(),
        B,
        C);

    return {loss, row_max, inv_sum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_ce_forward", &softmax_ce_forward,
          "Mean softmax cross-entropy forward (CUDA)");
}
