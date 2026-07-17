#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <climits>
#include <vector>

namespace {

constexpr int kTopK = 8;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;

__device__ __forceinline__ bool comes_before(
        float lhs_value, int lhs_index, float rhs_value, int rhs_index) {
    const bool lhs_nan = isnan(lhs_value);
    const bool rhs_nan = isnan(rhs_value);
    if (lhs_nan != rhs_nan) {
        return lhs_nan;
    }
    if (lhs_value != rhs_value) {
        return lhs_value > rhs_value;
    }
    return lhs_index < rhs_index;
}

__device__ __forceinline__ void swap_pair(
        float& lhs_value, int& lhs_index, float& rhs_value, int& rhs_index) {
    const float value = lhs_value;
    const int index = lhs_index;
    lhs_value = rhs_value;
    lhs_index = rhs_index;
    rhs_value = value;
    rhs_index = index;
}

__device__ __forceinline__ void warp_best(float& value, int& index) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(mask, value, offset);
        const int other_index = __shfl_down_sync(mask, index, offset);
        if (comes_before(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
    value = __shfl_sync(mask, value, 0);
    index = __shfl_sync(mask, index, 0);
}

__global__ __launch_bounds__(kThreads) void topk_forward_kernel(
        const float* __restrict__ X,
        float* __restrict__ values,
        int* __restrict__ indices,
        int N,
        int D) {
    const int row_index = blockIdx.x;
    if (row_index >= N) {
        return;
    }

    const int thread = threadIdx.x;
    const int lane = thread & (kWarpSize - 1);
    const int warp = thread / kWarpSize;
    const float* row = X + static_cast<size_t>(row_index) * D;

    float local_values[kTopK];
    int local_indices[kTopK];
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        local_values[rank] = -CUDART_INF_F;
        local_indices[rank] = INT_MAX;
    }

    for (int column = thread; column < D; column += kThreads) {
        const float value = row[column];
        if (!comes_before(value, column, local_values[kTopK - 1],
                          local_indices[kTopK - 1])) {
            continue;
        }
        local_values[kTopK - 1] = value;
        local_indices[kTopK - 1] = column;
#pragma unroll
        for (int rank = kTopK - 1; rank > 0; --rank) {
            if (!comes_before(local_values[rank], local_indices[rank],
                              local_values[rank - 1], local_indices[rank - 1])) {
                break;
            }
            swap_pair(local_values[rank], local_indices[rank],
                      local_values[rank - 1], local_indices[rank - 1]);
        }
    }

    __shared__ float warp_values[kWarps * kTopK];
    __shared__ int warp_indices[kWarps * kTopK];

    int cursor = 0;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        float candidate_value = cursor < kTopK ? local_values[cursor] : -CUDART_INF_F;
        int candidate_index = cursor < kTopK ? local_indices[cursor] : INT_MAX;
        warp_best(candidate_value, candidate_index);
        if (lane == 0) {
            warp_values[warp * kTopK + rank] = candidate_value;
            warp_indices[warp * kTopK + rank] = candidate_index;
        }
        if (cursor < kTopK && local_indices[cursor] == candidate_index) {
            ++cursor;
        }
    }
    __syncthreads();

    if (warp == 0) {
        float first_value = warp_values[lane];
        int first_index = warp_indices[lane];
        float second_value = warp_values[lane + kWarpSize];
        int second_index = warp_indices[lane + kWarpSize];
        if (comes_before(second_value, second_index, first_value, first_index)) {
            swap_pair(first_value, first_index, second_value, second_index);
        }

        int final_cursor = 0;
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            float candidate_value = final_cursor == 0
                ? first_value
                : (final_cursor == 1 ? second_value : -CUDART_INF_F);
            int candidate_index = final_cursor == 0
                ? first_index
                : (final_cursor == 1 ? second_index : INT_MAX);
            warp_best(candidate_value, candidate_index);
            if (lane == 0) {
                const size_t output = static_cast<size_t>(row_index) * kTopK + rank;
                values[output] = candidate_value;
                indices[output] = candidate_index;
            }
            const int own_index = final_cursor == 0
                ? first_index
                : (final_cursor == 1 ? second_index : INT_MAX);
            if (final_cursor < 2 && own_index == candidate_index) {
                ++final_cursor;
            }
        }
    }
}

__global__ __launch_bounds__(kThreads) void topk_backward_kernel(
        const float* __restrict__ grad_values,
        const int* __restrict__ indices,
        float* __restrict__ grad_X,
        int N,
        int D) {
    const int row_index = blockIdx.x;
    if (row_index >= N) {
        return;
    }

    float* grad_row = grad_X + static_cast<size_t>(row_index) * D;
    for (int column = threadIdx.x; column < D; column += kThreads) {
        grad_row[column] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x < kTopK) {
        const size_t offset = static_cast<size_t>(row_index) * kTopK + threadIdx.x;
        grad_row[indices[offset]] = grad_values[offset];
    }
}

}  // namespace

std::vector<torch::Tensor> topk_forward(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must have dtype float32");
    TORCH_CHECK(X.dim() == 2, "X must have shape [N, D]");
    TORCH_CHECK(X.size(1) >= kTopK, "D must be at least 8");

    const c10::cuda::CUDAGuard device_guard(X.device());
    X = X.contiguous();
    const int64_t N64 = X.size(0);
    const int64_t D64 = X.size(1);
    TORCH_CHECK(N64 <= INT_MAX && D64 <= INT_MAX, "N and D must fit in int32");
    const int N = static_cast<int>(N64);
    const int D = static_cast<int>(D64);

    auto values = torch::empty({N64, kTopK}, X.options());
    auto indices = torch::empty(
        {N64, kTopK}, X.options().dtype(torch::kInt32).requires_grad(false));
    if (N > 0) {
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        topk_forward_kernel<<<N, kThreads, 0, stream>>>(
            X.data_ptr<float>(), values.data_ptr<float>(), indices.data_ptr<int>(), N, D);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {values, indices};
}

torch::Tensor topk_backward(
        torch::Tensor grad_values, torch::Tensor indices, int64_t input_width) {
    TORCH_CHECK(grad_values.is_cuda() && indices.is_cuda(),
                "grad_values and indices must be CUDA tensors");
    TORCH_CHECK(grad_values.scalar_type() == torch::kFloat32,
                "grad_values must have dtype float32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32,
                "indices must have dtype int32");
    TORCH_CHECK(grad_values.dim() == 2 && grad_values.size(1) == kTopK,
                "grad_values must have shape [N, 8]");
    TORCH_CHECK(indices.sizes() == grad_values.sizes(),
                "indices must have the same shape as grad_values");
    TORCH_CHECK(input_width >= kTopK && input_width <= INT_MAX,
                "input_width must fit in int32 and be at least 8");

    const c10::cuda::CUDAGuard device_guard(grad_values.device());
    grad_values = grad_values.contiguous();
    indices = indices.contiguous();
    const int64_t N64 = grad_values.size(0);
    TORCH_CHECK(N64 <= INT_MAX, "N must fit in int32");
    const int N = static_cast<int>(N64);
    const int D = static_cast<int>(input_width);
    auto grad_X = torch::empty({N64, input_width}, grad_values.options());
    if (N > 0) {
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        topk_backward_kernel<<<N, kThreads, 0, stream>>>(
            grad_values.data_ptr<float>(), indices.data_ptr<int>(),
            grad_X.data_ptr<float>(), N, D);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return grad_X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("topk_forward", &topk_forward, "Row-wise Top-K forward (CUDA)");
    module.def("topk_backward", &topk_backward, "Row-wise Top-K backward (CUDA)");
}
