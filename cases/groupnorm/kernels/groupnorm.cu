#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[kWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < kWarps ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warp_sum(value);
    }
    if (threadIdx.x == 0) {
        warp_sums[0] = value;
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void groupnorm_forward_vec4_kernel(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean_output,
    float* __restrict__ rstd_output,
    int groups,
    int channels_per_group,
    int spatial,
    float eps) {
    const int row = blockIdx.x;
    const int group = row % groups;
    const int spatial4 = spatial >> 2;
    const int group4 = channels_per_group * spatial4;
    const std::int64_t group_offset4 = static_cast<std::int64_t>(row) * group4;
    const float4* X4 = reinterpret_cast<const float4*>(X) + group_offset4;

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int index4 = threadIdx.x; index4 < group4; index4 += kThreads) {
        const float4 value = X4[index4];
        local_sum += value.x + value.y + value.z + value.w;
        local_sq += value.x * value.x + value.y * value.y +
                    value.z * value.z + value.w * value.w;
    }
    const float sum = block_sum(local_sum);
    const float sum_sq = block_sum(local_sq);

    __shared__ float shared_mean;
    __shared__ float shared_rstd;
    if (threadIdx.x == 0) {
        const float inv_count = 1.0f / static_cast<float>(group4 * 4);
        const float mean = sum * inv_count;
        const float variance = fmaxf(sum_sq * inv_count - mean * mean, 0.0f);
        shared_mean = mean;
        shared_rstd = rsqrtf(variance + eps);
        mean_output[row] = mean;
        rstd_output[row] = shared_rstd;
    }
    __syncthreads();

    float4* output4 = reinterpret_cast<float4*>(output) + group_offset4;
    const int channel_base = group * channels_per_group;
    for (int index4 = threadIdx.x; index4 < group4; index4 += kThreads) {
        const float4 value = X4[index4];
        const int channel = channel_base + index4 / spatial4;
        const float scale = gamma[channel];
        const float bias = beta[channel];
        float4 result;
        result.x = (value.x - shared_mean) * shared_rstd * scale + bias;
        result.y = (value.y - shared_mean) * shared_rstd * scale + bias;
        result.z = (value.z - shared_mean) * shared_rstd * scale + bias;
        result.w = (value.w - shared_mean) * shared_rstd * scale + bias;
        output4[index4] = result;
    }
}

__global__ void groupnorm_forward_scalar_kernel(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean_output,
    float* __restrict__ rstd_output,
    int groups,
    int channels_per_group,
    int spatial,
    float eps) {
    const int row = blockIdx.x;
    const int group = row % groups;
    const int group_size = channels_per_group * spatial;
    const std::int64_t offset = static_cast<std::int64_t>(row) * group_size;

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int index = threadIdx.x; index < group_size; index += kThreads) {
        const float value = X[offset + index];
        local_sum += value;
        local_sq += value * value;
    }
    const float sum = block_sum(local_sum);
    const float sum_sq = block_sum(local_sq);

    __shared__ float shared_mean;
    __shared__ float shared_rstd;
    if (threadIdx.x == 0) {
        const float inv_count = 1.0f / static_cast<float>(group_size);
        const float mean = sum * inv_count;
        const float variance = fmaxf(sum_sq * inv_count - mean * mean, 0.0f);
        shared_mean = mean;
        shared_rstd = rsqrtf(variance + eps);
        mean_output[row] = mean;
        rstd_output[row] = shared_rstd;
    }
    __syncthreads();

    const int channel_base = group * channels_per_group;
    for (int index = threadIdx.x; index < group_size; index += kThreads) {
        const int channel = channel_base + index / spatial;
        output[offset + index] =
            (X[offset + index] - shared_mean) * shared_rstd * gamma[channel] +
            beta[channel];
    }
}

__global__ void groupnorm_backward_x_vec4_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_X,
    int groups,
    int channels_per_group,
    int spatial) {
    const int row = blockIdx.x;
    const int group = row % groups;
    const int spatial4 = spatial >> 2;
    const int group4 = channels_per_group * spatial4;
    const std::int64_t offset4 = static_cast<std::int64_t>(row) * group4;
    const float4* X4 = reinterpret_cast<const float4*>(X) + offset4;
    const float4* grad4 = reinterpret_cast<const float4*>(grad_output) + offset4;
    const int channel_base = group * channels_per_group;
    const float row_mean = mean[row];
    const float row_rstd = rstd[row];

    float local_dy = 0.0f;
    float local_dy_xhat = 0.0f;
    for (int index4 = threadIdx.x; index4 < group4; index4 += kThreads) {
        const float4 value = X4[index4];
        const float4 grad = grad4[index4];
        const float scale = gamma[channel_base + index4 / spatial4];
        const float x0 = (value.x - row_mean) * row_rstd;
        const float x1 = (value.y - row_mean) * row_rstd;
        const float x2 = (value.z - row_mean) * row_rstd;
        const float x3 = (value.w - row_mean) * row_rstd;
        const float d0 = grad.x * scale;
        const float d1 = grad.y * scale;
        const float d2 = grad.z * scale;
        const float d3 = grad.w * scale;
        local_dy += d0 + d1 + d2 + d3;
        local_dy_xhat += d0 * x0 + d1 * x1 + d2 * x2 + d3 * x3;
    }
    const float sum_dy = block_sum(local_dy);
    const float sum_dy_xhat = block_sum(local_dy_xhat);
    const float inv_count = 1.0f / static_cast<float>(group4 * 4);
    const float mean_dy = sum_dy * inv_count;
    const float mean_dy_xhat = sum_dy_xhat * inv_count;

    float4* grad_X4 = reinterpret_cast<float4*>(grad_X) + offset4;
    for (int index4 = threadIdx.x; index4 < group4; index4 += kThreads) {
        const float4 value = X4[index4];
        const float4 grad = grad4[index4];
        const float scale = gamma[channel_base + index4 / spatial4];
        const float x0 = (value.x - row_mean) * row_rstd;
        const float x1 = (value.y - row_mean) * row_rstd;
        const float x2 = (value.z - row_mean) * row_rstd;
        const float x3 = (value.w - row_mean) * row_rstd;
        const float d0 = grad.x * scale;
        const float d1 = grad.y * scale;
        const float d2 = grad.z * scale;
        const float d3 = grad.w * scale;
        float4 result;
        result.x = row_rstd * (d0 - mean_dy - x0 * mean_dy_xhat);
        result.y = row_rstd * (d1 - mean_dy - x1 * mean_dy_xhat);
        result.z = row_rstd * (d2 - mean_dy - x2 * mean_dy_xhat);
        result.w = row_rstd * (d3 - mean_dy - x3 * mean_dy_xhat);
        grad_X4[index4] = result;
    }
}

__global__ void groupnorm_backward_x_scalar_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_X,
    int groups,
    int channels_per_group,
    int spatial) {
    const int row = blockIdx.x;
    const int group = row % groups;
    const int group_size = channels_per_group * spatial;
    const std::int64_t offset = static_cast<std::int64_t>(row) * group_size;
    const int channel_base = group * channels_per_group;
    const float row_mean = mean[row];
    const float row_rstd = rstd[row];

    float local_dy = 0.0f;
    float local_dy_xhat = 0.0f;
    for (int index = threadIdx.x; index < group_size; index += kThreads) {
        const float xhat = (X[offset + index] - row_mean) * row_rstd;
        const float dxhat =
            grad_output[offset + index] * gamma[channel_base + index / spatial];
        local_dy += dxhat;
        local_dy_xhat += dxhat * xhat;
    }
    const float sum_dy = block_sum(local_dy);
    const float sum_dy_xhat = block_sum(local_dy_xhat);
    const float inv_count = 1.0f / static_cast<float>(group_size);
    const float mean_dy = sum_dy * inv_count;
    const float mean_dy_xhat = sum_dy_xhat * inv_count;

    for (int index = threadIdx.x; index < group_size; index += kThreads) {
        const float xhat = (X[offset + index] - row_mean) * row_rstd;
        const float dxhat =
            grad_output[offset + index] * gamma[channel_base + index / spatial];
        grad_X[offset + index] =
            row_rstd * (dxhat - mean_dy - xhat * mean_dy_xhat);
    }
}

__global__ void groupnorm_backward_param_vec4_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ X,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int batch,
    int channels,
    int groups,
    int spatial) {
    const int channel = blockIdx.x;
    const int group = channel / (channels / groups);
    const int spatial4 = spatial >> 2;
    float local_gamma = 0.0f;
    float local_beta = 0.0f;
    for (int item = threadIdx.x; item < batch * spatial4; item += kThreads) {
        const int n = item / spatial4;
        const int index4 = item - n * spatial4;
        const std::int64_t offset =
            (static_cast<std::int64_t>(n) * channels + channel) * spatial;
        const float4 value = reinterpret_cast<const float4*>(X + offset)[index4];
        const float4 grad =
            reinterpret_cast<const float4*>(grad_output + offset)[index4];
        const int row = n * groups + group;
        const float row_mean = mean[row];
        const float row_rstd = rstd[row];
        local_gamma +=
            grad.x * ((value.x - row_mean) * row_rstd) +
            grad.y * ((value.y - row_mean) * row_rstd) +
            grad.z * ((value.z - row_mean) * row_rstd) +
            grad.w * ((value.w - row_mean) * row_rstd);
        local_beta += grad.x + grad.y + grad.z + grad.w;
    }
    const float gamma_sum = block_sum(local_gamma);
    const float beta_sum = block_sum(local_beta);
    if (threadIdx.x == 0) {
        grad_gamma[channel] = gamma_sum;
        grad_beta[channel] = beta_sum;
    }
}

__global__ void groupnorm_backward_param_scalar_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ X,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int batch,
    int channels,
    int groups,
    int spatial) {
    const int channel = blockIdx.x;
    const int group = channel / (channels / groups);
    float local_gamma = 0.0f;
    float local_beta = 0.0f;
    for (int item = threadIdx.x; item < batch * spatial; item += kThreads) {
        const int n = item / spatial;
        const int s = item - n * spatial;
        const std::int64_t index =
            (static_cast<std::int64_t>(n) * channels + channel) * spatial + s;
        const float grad = grad_output[index];
        const int row = n * groups + group;
        local_gamma += grad * ((X[index] - mean[row]) * rstd[row]);
        local_beta += grad;
    }
    const float gamma_sum = block_sum(local_gamma);
    const float beta_sum = block_sum(local_beta);
    if (threadIdx.x == 0) {
        grad_gamma[channel] = gamma_sum;
        grad_beta[channel] = beta_sum;
    }
}

void check_forward(
    const torch::Tensor& X,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int groups) {
    TORCH_CHECK(X.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    gamma.scalar_type() == torch::kFloat32 &&
                    beta.scalar_type() == torch::kFloat32,
                "all inputs must be float32");
    TORCH_CHECK(X.dim() == 4, "X must have shape [N,C,H,W]");
    TORCH_CHECK(X.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1 &&
                    gamma.size(0) == X.size(1) && beta.size(0) == X.size(1),
                "gamma and beta must have shape [C]");
    TORCH_CHECK(groups > 0 && X.size(1) % groups == 0,
                "groups must be positive and divide C");
    TORCH_CHECK(X.size(0) > 0 && X.size(2) > 0 && X.size(3) > 0,
                "N, H, and W must be positive");
}

void check_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& X,
    const torch::Tensor& gamma,
    const torch::Tensor& mean,
    const torch::Tensor& rstd,
    int groups) {
    TORCH_CHECK(grad_output.is_cuda() && X.is_cuda() && gamma.is_cuda() &&
                    mean.is_cuda() && rstd.is_cuda(),
                "all backward inputs must be CUDA tensors");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32 &&
                    X.scalar_type() == torch::kFloat32 &&
                    gamma.scalar_type() == torch::kFloat32 &&
                    mean.scalar_type() == torch::kFloat32 &&
                    rstd.scalar_type() == torch::kFloat32,
                "all backward inputs must be float32");
    TORCH_CHECK(X.dim() == 4 && grad_output.sizes() == X.sizes(),
                "grad_output shape must match NCHW X");
    TORCH_CHECK(grad_output.is_contiguous() && X.is_contiguous() &&
                    gamma.is_contiguous() && mean.is_contiguous() && rstd.is_contiguous(),
                "all backward inputs must be contiguous");
    TORCH_CHECK(groups > 0 && X.size(1) % groups == 0,
                "groups must be positive and divide C");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == X.size(1),
                "gamma must have shape [C]");
    TORCH_CHECK(mean.dim() == 1 && rstd.dim() == 1 &&
                    mean.numel() == X.size(0) * groups &&
                    rstd.numel() == X.size(0) * groups,
                "saved statistics must have shape [N*groups]");
}

}  // namespace

std::vector<torch::Tensor> groupnorm_forward(
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t groups,
    double eps) {
    check_forward(X, gamma, beta, static_cast<int>(groups));
    const int batch = static_cast<int>(X.size(0));
    const int channels = static_cast<int>(X.size(1));
    const int spatial = static_cast<int>(X.size(2) * X.size(3));
    const int group_count = static_cast<int>(groups);
    const int channels_per_group = channels / group_count;
    auto output = torch::empty_like(X);
    auto mean = torch::empty({batch * group_count}, X.options());
    auto rstd = torch::empty({batch * group_count}, X.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blocks = batch * group_count;
    if ((spatial & 3) == 0) {
        groupnorm_forward_vec4_kernel<<<blocks, kThreads, 0, stream>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            group_count, channels_per_group, spatial, static_cast<float>(eps));
    } else {
        groupnorm_forward_scalar_kernel<<<blocks, kThreads, 0, stream>>>(
            X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
            group_count, channels_per_group, spatial, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, mean, rstd};
}

std::vector<torch::Tensor> groupnorm_backward(
    torch::Tensor grad_output,
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor mean,
    torch::Tensor rstd,
    int64_t groups) {
    check_backward(grad_output, X, gamma, mean, rstd, static_cast<int>(groups));
    const int batch = static_cast<int>(X.size(0));
    const int channels = static_cast<int>(X.size(1));
    const int spatial = static_cast<int>(X.size(2) * X.size(3));
    const int group_count = static_cast<int>(groups);
    const int channels_per_group = channels / group_count;
    auto grad_X = torch::empty_like(X);
    auto grad_gamma = torch::empty_like(gamma);
    auto grad_beta = torch::empty_like(gamma);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blocks = batch * group_count;
    if ((spatial & 3) == 0) {
        groupnorm_backward_x_vec4_kernel<<<blocks, kThreads, 0, stream>>>(
            grad_output.data_ptr<float>(), X.data_ptr<float>(), gamma.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(), grad_X.data_ptr<float>(),
            group_count, channels_per_group, spatial);
        groupnorm_backward_param_vec4_kernel<<<channels, kThreads, 0, stream>>>(
            grad_output.data_ptr<float>(), X.data_ptr<float>(), mean.data_ptr<float>(),
            rstd.data_ptr<float>(), grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
            batch, channels, group_count, spatial);
    } else {
        groupnorm_backward_x_scalar_kernel<<<blocks, kThreads, 0, stream>>>(
            grad_output.data_ptr<float>(), X.data_ptr<float>(), gamma.data_ptr<float>(),
            mean.data_ptr<float>(), rstd.data_ptr<float>(), grad_X.data_ptr<float>(),
            group_count, channels_per_group, spatial);
        groupnorm_backward_param_scalar_kernel<<<channels, kThreads, 0, stream>>>(
            grad_output.data_ptr<float>(), X.data_ptr<float>(), mean.data_ptr<float>(),
            rstd.data_ptr<float>(), grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
            batch, channels, group_count, spatial);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_X, grad_gamma, grad_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("groupnorm_forward", &groupnorm_forward, "GroupNorm forward (CUDA)");
    module.def("groupnorm_backward", &groupnorm_backward, "GroupNorm backward (CUDA)");
}
