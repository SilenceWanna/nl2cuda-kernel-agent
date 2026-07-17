#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kMaxBlocks = 4096;
constexpr int kBackwardThreads = 256;
constexpr int kBackwardWarpsPerBlock = kBackwardThreads / kWarpSize;
constexpr int kBackwardMaxBlocks = 4096;

__global__ void scatter_add_forward_kernel(
    const float* __restrict__ X,
    const int64_t* __restrict__ idx,
    float* __restrict__ Y,
    int64_t N,
    int64_t D,
    int64_t S) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int64_t first_source =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    const int64_t source_stride =
        static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

    for (int64_t i = first_source; i < N; i += source_stride) {
        int64_t segment = 0;
        if (lane == 0) {
            segment = idx[i];
        }
        segment = __shfl_sync(0xffffffffu, segment, 0);
        if (segment < 0 || segment >= S) {
            continue;
        }

        const float* source = X + i * D;
        float* destination = Y + segment * D;
        for (int64_t d = lane; d < D; d += kWarpSize) {
            atomicAdd(destination + d, source[d]);
        }
    }
}

__global__ void scatter_add_backward_kernel(
    const float* __restrict__ grad_Y,
    const int64_t* __restrict__ idx,
    float* __restrict__ grad_X,
    int64_t N,
    int64_t D,
    int64_t S) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int64_t first_source =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    const int64_t source_stride =
        static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

    for (int64_t i = first_source; i < N; i += source_stride) {
        int64_t segment = 0;
        if (lane == 0) {
            segment = idx[i];
        }
        segment = __shfl_sync(0xffffffffu, segment, 0);

        float* destination = grad_X + i * D;
        if (segment < 0 || segment >= S) {
            for (int64_t d = lane; d < D; d += kWarpSize) {
                destination[d] = 0.0f;
            }
            continue;
        }

        const float* source = grad_Y + segment * D;
        if ((D & 3) == 0) {
            const int64_t vector_count = D / 4;
            const float4* source4 = reinterpret_cast<const float4*>(source);
            float4* destination4 = reinterpret_cast<float4*>(destination);
            for (int64_t vector = lane; vector < vector_count; vector += kWarpSize) {
                destination4[vector] = source4[vector];
            }
        } else {
            for (int64_t d = lane; d < D; d += kWarpSize) {
                destination[d] = source[d];
            }
        }
    }
}

template <int FeatureDim>
__global__ void __launch_bounds__(kBackwardThreads, 6)
scatter_add_backward_fixed_kernel(
    const float* __restrict__ grad_Y,
    const int64_t* __restrict__ idx,
    float* __restrict__ grad_X,
    int N) {
    static_assert(FeatureDim % 4 == 0, "FeatureDim must support float4 access");
    constexpr int kVectorsPerRow = FeatureDim / 4;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int first_source = blockIdx.x * kBackwardWarpsPerBlock + warp_in_block;
    const int source_stride = gridDim.x * kBackwardWarpsPerBlock;

    for (int i = first_source; i < N; i += source_stride) {
        int segment = 0;
        if (lane == 0) {
            segment = static_cast<int>(idx[i]);
        }
        segment = __shfl_sync(0xffffffffu, segment, 0);
        const float4* source = reinterpret_cast<const float4*>(grad_Y) +
            static_cast<int64_t>(segment) * kVectorsPerRow;
        float4* destination = reinterpret_cast<float4*>(grad_X) +
            static_cast<int64_t>(i) * kVectorsPerRow;

#pragma unroll
        for (int vector = lane; vector < kVectorsPerRow; vector += kWarpSize) {
            destination[vector] = source[vector];
        }
    }
}

void check_indices(const torch::Tensor& idx, int64_t expected_size) {
    TORCH_CHECK(idx.is_cuda(), "idx must be a CUDA tensor");
    TORCH_CHECK(idx.scalar_type() == torch::kInt64, "idx must be int64");
    TORCH_CHECK(idx.is_contiguous(), "idx must be contiguous");
    TORCH_CHECK(idx.dim() == 1, "idx must have shape [N]");
    TORCH_CHECK(idx.size(0) == expected_size, "idx length must equal N");
}

int launch_blocks(int64_t N) {
    const int64_t required = (N + kWarpsPerBlock - 1) / kWarpsPerBlock;
    return static_cast<int>(std::min<int64_t>(required, kMaxBlocks));
}

int launch_backward_blocks(int64_t N) {
    const int64_t required =
        (N + kBackwardWarpsPerBlock - 1) / kBackwardWarpsPerBlock;
    return static_cast<int>(std::min<int64_t>(required, kBackwardMaxBlocks));
}

}  // namespace

torch::Tensor scatter_add_forward(
    torch::Tensor X,
    torch::Tensor idx,
    int64_t segments) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(X.dim() == 2, "X must have shape [N, D]");
    TORCH_CHECK(segments > 0, "S must be positive");

    const int64_t N = X.size(0);
    const int64_t D = X.size(1);
    TORCH_CHECK(D > 0, "D must be positive");
    check_indices(idx, N);

    auto Y = torch::empty({segments, D}, X.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        Y.data_ptr<float>(),
        0,
        static_cast<size_t>(segments * D) * sizeof(float),
        stream));

    if (N > 0) {
        scatter_add_forward_kernel<<<launch_blocks(N), kThreads, 0, stream>>>(
            X.data_ptr<float>(),
            idx.data_ptr<int64_t>(),
            Y.data_ptr<float>(),
            N,
            D,
            segments);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return Y;
}

torch::Tensor scatter_add_backward(
    torch::Tensor grad_Y,
    torch::Tensor idx) {
    TORCH_CHECK(grad_Y.is_cuda(), "grad_Y must be a CUDA tensor");
    TORCH_CHECK(
        grad_Y.scalar_type() == torch::kFloat32,
        "grad_Y must be float32");
    TORCH_CHECK(grad_Y.is_contiguous(), "grad_Y must be contiguous");
    TORCH_CHECK(grad_Y.dim() == 2, "grad_Y must have shape [S, D]");

    const int64_t N = idx.size(0);
    const int64_t D = grad_Y.size(1);
    const int64_t S = grad_Y.size(0);
    check_indices(idx, N);

    auto grad_X = torch::empty({N, D}, grad_Y.options());
    if (N > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        if (D == 128 && N <= 2147483647LL && S <= 2147483647LL) {
            scatter_add_backward_fixed_kernel<128>
                <<<launch_backward_blocks(N), kBackwardThreads, 0, stream>>>(
                    grad_Y.data_ptr<float>(),
                    idx.data_ptr<int64_t>(),
                    grad_X.data_ptr<float>(),
                    static_cast<int>(N));
        } else {
            scatter_add_backward_kernel<<<launch_blocks(N), kThreads, 0, stream>>>(
                grad_Y.data_ptr<float>(),
                idx.data_ptr<int64_t>(),
                grad_X.data_ptr<float>(),
                N,
                D,
                S);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return grad_X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "scatter_add_forward",
        &scatter_add_forward,
        "Scatter-add forward (CUDA)");
    module.def(
        "scatter_add_backward",
        &scatter_add_backward,
        "Scatter-add backward (CUDA)");
}
