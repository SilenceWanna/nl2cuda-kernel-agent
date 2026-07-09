#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cfloat>
#include <cmath>
#include <cstdint>


#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)


template <typename scalar_t>
__global__ void softmax_ce_forward_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ target,
    float* __restrict__ losses,
    int64_t n,
    int64_t c) {
    extern __shared__ float smem[];

    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= n) {
        return;
    }

    const int64_t base = row * c;

    float local_max = -FLT_MAX;
    for (int64_t col = tid; col < c; col += blockDim.x) {
        const float v = static_cast<float>(logits[base + col]);
        local_max = fmaxf(local_max, v);
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = smem[0];

    float local_sum = 0.0f;
    for (int64_t col = tid; col < c; col += blockDim.x) {
        const float v = static_cast<float>(logits[base + col]);
        local_sum += expf(v - row_max);
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
        const int64_t y = target[row];
        const float correct = static_cast<float>(logits[base + y]);
        losses[row] = row_max + logf(smem[0]) - correct;
    }
}


torch::Tensor softmax_ce_forward(torch::Tensor logits, torch::Tensor target) {
    CHECK_INPUT(logits);
    CHECK_INPUT(target);

    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    TORCH_CHECK(target.dim() == 1, "target must be a 1D tensor");
    TORCH_CHECK(target.scalar_type() == at::kLong, "target must have dtype torch.long");
    TORCH_CHECK(logits.size(0) == target.size(0), "target length must match logits batch size");
    TORCH_CHECK(logits.size(0) > 0, "logits batch size must be positive");
    TORCH_CHECK(logits.size(1) > 0, "logits class dimension must be positive");

    const auto n = logits.size(0);
    const auto c = logits.size(1);

    const c10::cuda::CUDAGuard device_guard(logits.device());

    auto losses = torch::empty({n}, logits.options().dtype(torch::kFloat32));

    constexpr int threads = 256;
    const size_t shared = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "softmax_ce_forward_cuda", [&] {
        softmax_ce_forward_kernel<scalar_t>
            <<<n, threads, shared, at::cuda::getCurrentCUDAStream()>>>(
                logits.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(),
                losses.data_ptr<float>(),
                n,
                c);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return losses.mean();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_ce_forward", &softmax_ce_forward, "Softmax cross entropy forward");
}
