#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kFastD = 1024;
constexpr int kParamRowTile = 32;

struct WelfordState {
    float mean;
    float m2;
    int count;
};

__device__ __forceinline__ WelfordState welford_empty() {
    return {0.0f, 0.0f, 0};
}

__device__ __forceinline__ void welford_update(WelfordState& state, float value) {
    const int next_count = state.count + 1;
    const float delta = value - state.mean;
    state.mean += delta / static_cast<float>(next_count);
    const float delta2 = value - state.mean;
    state.m2 += delta * delta2;
    state.count = next_count;
}

__device__ __forceinline__ WelfordState welford_combine(
    WelfordState left,
    WelfordState right) {
    if (left.count == 0) {
        return right;
    }
    if (right.count == 0) {
        return left;
    }

    const int count = left.count + right.count;
    const float delta = right.mean - left.mean;
    const float right_weight = static_cast<float>(right.count) /
                               static_cast<float>(count);
    const float cross_weight =
        static_cast<float>(left.count) * static_cast<float>(right.count) /
        static_cast<float>(count);
    left.mean += delta * right_weight;
    left.m2 += right.m2 + delta * delta * cross_weight;
    left.count = count;
    return left;
}

__device__ __forceinline__ WelfordState warp_reduce_welford(
    WelfordState state) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordState other;
        other.mean = __shfl_down_sync(0xffffffff, state.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, state.m2, offset);
        other.count = __shfl_down_sync(0xffffffff, state.count, offset);
        if (lane + offset < 32) {
            state = welford_combine(state, other);
        }
    }
    return state;
}

__device__ __forceinline__ WelfordState block_reduce_welford(
    WelfordState state) {
    __shared__ WelfordState warp_states[kWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    state = warp_reduce_welford(state);
    if (lane == 0) {
        warp_states[warp] = state;
    }
    __syncthreads();

    if (warp == 0) {
        state = lane < kWarps ? warp_states[lane] : welford_empty();
        state = warp_reduce_welford(state);
        if (lane == 0) {
            warp_states[0] = state;
        }
    }
    __syncthreads();
    return warp_states[0];
}

// All lanes start with equally sized chunks. This Welford merge specialization
// avoids divisions on the default D=1024 path.
__device__ __forceinline__ void reduce_equal_chunks(
    float& mean,
    float& m2,
    int logical_width,
    float initial_count) {
    const int lane = threadIdx.x & 31;
    float count = initial_count;
    for (int offset = logical_width >> 1; offset > 0; offset >>= 1) {
        const float other_mean = __shfl_down_sync(0xffffffff, mean, offset);
        const float other_m2 = __shfl_down_sync(0xffffffff, m2, offset);
        if (lane < offset) {
            const float delta = other_mean - mean;
            mean += delta * 0.5f;
            m2 += other_m2 + delta * delta * (count * 0.5f);
            count *= 2.0f;
        }
    }
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[kWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < kWarps ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
    }
    if (threadIdx.x == 0) {
        warp_sums[0] = value;
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void welford_forward_1024_kernel(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean_output,
    float* __restrict__ rstd_output,
    int rows,
    float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const std::int64_t row_offset = static_cast<std::int64_t>(row) * kFastD;
    const float4* X4 = reinterpret_cast<const float4*>(X + row_offset);
    const float4 values = X4[tid];

    WelfordState local = welford_empty();
    welford_update(local, values.x);
    welford_update(local, values.y);
    welford_update(local, values.z);
    welford_update(local, values.w);

    float local_mean = local.mean;
    float local_m2 = local.m2;
    reduce_equal_chunks(local_mean, local_m2, 32, 4.0f);

    __shared__ float warp_means[kWarps];
    __shared__ float warp_m2s[kWarps];
    __shared__ float row_mean_shared;
    __shared__ float row_rstd_shared;

    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_means[warp] = local_mean;
        warp_m2s[warp] = local_m2;
    }
    __syncthreads();

    if (warp == 0) {
        float mean = lane < kWarps ? warp_means[lane] : 0.0f;
        float m2 = lane < kWarps ? warp_m2s[lane] : 0.0f;
        reduce_equal_chunks(mean, m2, kWarps, 128.0f);
        if (lane == 0) {
            const float variance = m2 * (1.0f / static_cast<float>(kFastD));
            const float rstd = rsqrtf(variance + eps);
            row_mean_shared = mean;
            row_rstd_shared = rstd;
            mean_output[row] = mean;
            rstd_output[row] = rstd;
        }
    }
    __syncthreads();

    const float4 gamma_values = reinterpret_cast<const float4*>(gamma)[tid];
    const float4 beta_values = reinterpret_cast<const float4*>(beta)[tid];
    const float mean = row_mean_shared;
    const float rstd = row_rstd_shared;
    float4 result;
    result.x = (values.x - mean) * rstd * gamma_values.x + beta_values.x;
    result.y = (values.y - mean) * rstd * gamma_values.y + beta_values.y;
    result.z = (values.z - mean) * rstd * gamma_values.z + beta_values.z;
    result.w = (values.w - mean) * rstd * gamma_values.w + beta_values.w;
    reinterpret_cast<float4*>(output + row_offset)[tid] = result;
}

__global__ void welford_forward_kernel(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean_output,
    float* __restrict__ rstd_output,
    int rows,
    int D,
    float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const std::int64_t row_offset = static_cast<std::int64_t>(row) * D;
    const float* X_row = X + row_offset;
    WelfordState local = welford_empty();
    for (int column = tid; column < D; column += kThreads) {
        welford_update(local, X_row[column]);
    }

    const WelfordState total = block_reduce_welford(local);
    __shared__ float row_mean;
    __shared__ float row_rstd;
    if (tid == 0) {
        const float variance = total.m2 / static_cast<float>(D);
        row_mean = total.mean;
        row_rstd = rsqrtf(variance + eps);
        mean_output[row] = row_mean;
        rstd_output[row] = row_rstd;
    }
    __syncthreads();

    float* output_row = output + row_offset;
    for (int column = tid; column < D; column += kThreads) {
        const float normalized = (X_row[column] - row_mean) * row_rstd;
        output_row[column] = normalized * gamma[column] + beta[column];
    }
}

__global__ void welford_backward_x_1024_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_X,
    int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const std::int64_t row_offset = static_cast<std::int64_t>(row) * kFastD;
    const float4 X_values =
        reinterpret_cast<const float4*>(X + row_offset)[tid];
    const float4 grad_values =
        reinterpret_cast<const float4*>(grad_output + row_offset)[tid];
    const float4 gamma_values = reinterpret_cast<const float4*>(gamma)[tid];
    const float row_mean = mean[row];
    const float row_rstd = rstd[row];

    const float xhat0 = (X_values.x - row_mean) * row_rstd;
    const float xhat1 = (X_values.y - row_mean) * row_rstd;
    const float xhat2 = (X_values.z - row_mean) * row_rstd;
    const float xhat3 = (X_values.w - row_mean) * row_rstd;
    const float dxhat0 = grad_values.x * gamma_values.x;
    const float dxhat1 = grad_values.y * gamma_values.y;
    const float dxhat2 = grad_values.z * gamma_values.z;
    const float dxhat3 = grad_values.w * gamma_values.w;

    float sum_dxhat = dxhat0 + dxhat1 + dxhat2 + dxhat3;
    float sum_dxhat_xhat =
        dxhat0 * xhat0 + dxhat1 * xhat1 + dxhat2 * xhat2 + dxhat3 * xhat3;
    sum_dxhat = block_reduce_sum(sum_dxhat);
    sum_dxhat_xhat = block_reduce_sum(sum_dxhat_xhat);

    const float mean_dxhat = sum_dxhat * (1.0f / static_cast<float>(kFastD));
    const float mean_dxhat_xhat =
        sum_dxhat_xhat * (1.0f / static_cast<float>(kFastD));
    float4 grad_X_values;
    grad_X_values.x =
        row_rstd * (dxhat0 - mean_dxhat - xhat0 * mean_dxhat_xhat);
    grad_X_values.y =
        row_rstd * (dxhat1 - mean_dxhat - xhat1 * mean_dxhat_xhat);
    grad_X_values.z =
        row_rstd * (dxhat2 - mean_dxhat - xhat2 * mean_dxhat_xhat);
    grad_X_values.w =
        row_rstd * (dxhat3 - mean_dxhat - xhat3 * mean_dxhat_xhat);
    reinterpret_cast<float4*>(grad_X + row_offset)[tid] = grad_X_values;
}

__global__ void welford_backward_x_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_X,
    int rows,
    int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const std::int64_t row_offset = static_cast<std::int64_t>(row) * D;
    const float* X_row = X + row_offset;
    const float* grad_row = grad_output + row_offset;
    const float row_mean = mean[row];
    const float row_rstd = rstd[row];

    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (int column = tid; column < D; column += kThreads) {
        const float xhat = (X_row[column] - row_mean) * row_rstd;
        const float dxhat = grad_row[column] * gamma[column];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
    }
    sum_dxhat = block_reduce_sum(sum_dxhat);
    sum_dxhat_xhat = block_reduce_sum(sum_dxhat_xhat);

    const float inv_D = 1.0f / static_cast<float>(D);
    const float mean_dxhat = sum_dxhat * inv_D;
    const float mean_dxhat_xhat = sum_dxhat_xhat * inv_D;
    float* grad_X_row = grad_X + row_offset;
    for (int column = tid; column < D; column += kThreads) {
        const float xhat = (X_row[column] - row_mean) * row_rstd;
        const float dxhat = grad_row[column] * gamma[column];
        grad_X_row[column] =
            row_rstd * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
    }
}

__global__ void welford_backward_param_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ X,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int rows,
    int D) {
    const int column = blockIdx.x * kThreads + threadIdx.x;
    if (column >= D) {
        return;
    }

    const int row_begin = blockIdx.y * kParamRowTile;
    const int row_end = min(row_begin + kParamRowTile, rows);
    float gamma_sum = 0.0f;
    float beta_sum = 0.0f;
#pragma unroll
    for (int row = row_begin; row < row_end; ++row) {
        const std::int64_t index = static_cast<std::int64_t>(row) * D + column;
        const float grad = grad_output[index];
        const float xhat = (X[index] - mean[row]) * rstd[row];
        gamma_sum += grad * xhat;
        beta_sum += grad;
    }
    atomicAdd(grad_gamma + column, gamma_sum);
    atomicAdd(grad_beta + column, beta_sum);
}

void check_forward_inputs(
    const torch::Tensor& X,
    const torch::Tensor& gamma,
    const torch::Tensor& beta) {
    TORCH_CHECK(X.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    gamma.scalar_type() == torch::kFloat32 &&
                    beta.scalar_type() == torch::kFloat32,
                "all inputs must be float32");
    TORCH_CHECK(X.dim() == 3, "X must have shape [B,N,D]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1,
                "gamma and beta must have shape [D]");
    TORCH_CHECK(X.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(gamma.size(0) == X.size(2) && beta.size(0) == X.size(2),
                "gamma and beta lengths must equal D");
    TORCH_CHECK(X.size(2) > 0, "D must be positive");
}

void check_backward_inputs(
    const torch::Tensor& grad_output,
    const torch::Tensor& X,
    const torch::Tensor& gamma,
    const torch::Tensor& mean,
    const torch::Tensor& rstd) {
    TORCH_CHECK(grad_output.is_cuda() && X.is_cuda() && gamma.is_cuda() &&
                    mean.is_cuda() && rstd.is_cuda(),
                "all backward inputs must be CUDA tensors");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32 &&
                    X.scalar_type() == torch::kFloat32 &&
                    gamma.scalar_type() == torch::kFloat32 &&
                    mean.scalar_type() == torch::kFloat32 &&
                    rstd.scalar_type() == torch::kFloat32,
                "all backward inputs must be float32");
    TORCH_CHECK(grad_output.sizes() == X.sizes(),
                "grad_output shape must match X");
    TORCH_CHECK(grad_output.is_contiguous() && X.is_contiguous() &&
                    gamma.is_contiguous() && mean.is_contiguous() &&
                    rstd.is_contiguous(),
                "all backward inputs must be contiguous");
    TORCH_CHECK(X.dim() == 3, "X must have shape [B,N,D]");
    const std::int64_t rows = X.size(0) * X.size(1);
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == X.size(2),
                "gamma shape must be [D]");
    TORCH_CHECK(mean.dim() == 1 && rstd.dim() == 1 &&
                    mean.size(0) == rows && rstd.size(0) == rows,
                "saved statistics must have shape [B*N]");
}

}  // namespace

std::vector<torch::Tensor> welford_forward(
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    check_forward_inputs(X, gamma, beta);
    const int rows = static_cast<int>(X.size(0) * X.size(1));
    const int D = static_cast<int>(X.size(2));
    auto output = torch::empty_like(X);
    auto mean = torch::empty({rows}, X.options());
    auto rstd = torch::empty({rows}, X.options());

    if (rows == 0) {
        return {output, mean, rstd};
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (D == kFastD) {
        welford_forward_1024_kernel<<<rows, kThreads, 0, stream>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            rows, static_cast<float>(eps));
    } else {
        welford_forward_kernel<<<rows, kThreads, 0, stream>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            rows, D, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, mean, rstd};
}

std::vector<torch::Tensor> welford_backward(
    torch::Tensor grad_output,
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor mean,
    torch::Tensor rstd) {
    check_backward_inputs(grad_output, X, gamma, mean, rstd);
    const int rows = static_cast<int>(X.size(0) * X.size(1));
    const int D = static_cast<int>(X.size(2));
    auto grad_X = torch::empty_like(X);
    auto grad_gamma = torch::empty_like(gamma);
    auto grad_beta = torch::empty_like(gamma);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        grad_gamma.data_ptr<float>(), 0, D * sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        grad_beta.data_ptr<float>(), 0, D * sizeof(float), stream));

    if (rows == 0) {
        return {grad_X, grad_gamma, grad_beta};
    }

    if (D == kFastD) {
        welford_backward_x_1024_kernel<<<rows, kThreads, 0, stream>>>(
            grad_output.data_ptr<float>(), X.data_ptr<float>(), gamma.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(), grad_X.data_ptr<float>(),
            rows);
    } else {
        welford_backward_x_kernel<<<rows, kThreads, 0, stream>>>(
            grad_output.data_ptr<float>(), X.data_ptr<float>(), gamma.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(), grad_X.data_ptr<float>(),
            rows, D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 param_grid(
        (D + kThreads - 1) / kThreads,
        (rows + kParamRowTile - 1) / kParamRowTile);
    welford_backward_param_kernel<<<param_grid, kThreads, 0, stream>>>(
        grad_output.data_ptr<float>(), X.data_ptr<float>(), mean.data_ptr<float>(),
        rstd.data_ptr<float>(), grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
        rows, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_X, grad_gamma, grad_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("welford_forward", &welford_forward,
               "Welford LayerNorm forward (CUDA)");
    module.def("welford_backward", &welford_backward,
               "Welford LayerNorm backward (CUDA)");
}
