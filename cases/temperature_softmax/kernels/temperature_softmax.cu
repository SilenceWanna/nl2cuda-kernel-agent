// Temperature-scaled row-wise softmax, forward and backward (float32).
//
// Forward:
//   p_i = exp(x_i / temperature - max_j(x_j / temperature)) / sum_j exp(...)
// Backward:
//   dx_i = p_i * (g_i - sum_j(g_j * p_j)) / temperature
//
// The D=1024 path keeps one float4 per thread in registers so the forward
// reads scores once and writes probabilities once. Other row sizes use a
// general block-per-row fallback.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(kFullMask, value, offset));
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < (blockDim.x >> 5) ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
    }
    if (threadIdx.x == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_max(float value, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_max(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < (blockDim.x >> 5)
        ? shared[lane]
        : -3.402823466e+38F;
    if (warp == 0) {
        value = warp_reduce_max(value);
    }
    if (threadIdx.x == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

__global__ void temperature_softmax_forward_1024_kernel(
        const float* __restrict__ scores,
        float* __restrict__ probabilities,
        int batch,
        float inv_temperature) {
    const int row = blockIdx.x;
    if (row >= batch) {
        return;
    }

    __shared__ float shared[8];
    const float4* input = reinterpret_cast<const float4*>(scores + (size_t)row * 1024);
    float4* output = reinterpret_cast<float4*>(probabilities + (size_t)row * 1024);

    const float4 packed = input[threadIdx.x];
    float values[4] = {
        packed.x * inv_temperature,
        packed.y * inv_temperature,
        packed.z * inv_temperature,
        packed.w * inv_temperature,
    };
    float local_max = fmaxf(fmaxf(values[0], values[1]),
                            fmaxf(values[2], values[3]));
    const float row_max = block_reduce_max(local_max, shared);

    float exponentials[4];
    exponentials[0] = expf(values[0] - row_max);
    exponentials[1] = expf(values[1] - row_max);
    exponentials[2] = expf(values[2] - row_max);
    exponentials[3] = expf(values[3] - row_max);
    const float local_sum = exponentials[0] + exponentials[1]
                          + exponentials[2] + exponentials[3];
    const float inverse_sum = 1.0f / block_reduce_sum(local_sum, shared);

    output[threadIdx.x] = make_float4(
        exponentials[0] * inverse_sum,
        exponentials[1] * inverse_sum,
        exponentials[2] * inverse_sum,
        exponentials[3] * inverse_sum);
}

__global__ void temperature_softmax_forward_generic_kernel(
        const float* __restrict__ scores,
        float* __restrict__ probabilities,
        int batch,
        int width,
        float inv_temperature) {
    const int row = blockIdx.x;
    if (row >= batch) {
        return;
    }

    __shared__ float shared[8];
    const float* input = scores + (size_t)row * width;
    float* output = probabilities + (size_t)row * width;

    float local_max = -3.402823466e+38F;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        local_max = fmaxf(local_max, input[column] * inv_temperature);
    }
    const float row_max = block_reduce_max(local_max, shared);

    float local_sum = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        local_sum += expf(input[column] * inv_temperature - row_max);
    }
    const float inverse_sum = 1.0f / block_reduce_sum(local_sum, shared);

    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        output[column] = expf(input[column] * inv_temperature - row_max) * inverse_sum;
    }
}

__global__ void temperature_softmax_backward_1024_kernel(
        const float* __restrict__ probabilities,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_scores,
        int batch,
        float inv_temperature) {
    const int row = blockIdx.x;
    if (row >= batch) {
        return;
    }

    __shared__ float shared[8];
    const float4* probabilities4 = reinterpret_cast<const float4*>(
        probabilities + (size_t)row * 1024);
    const float4* grad_output4 = reinterpret_cast<const float4*>(
        grad_output + (size_t)row * 1024);
    float4* grad_scores4 = reinterpret_cast<float4*>(
        grad_scores + (size_t)row * 1024);

    const float4 p = probabilities4[threadIdx.x];
    const float4 g = grad_output4[threadIdx.x];
    const float local_dot = p.x * g.x + p.y * g.y + p.z * g.z + p.w * g.w;
    const float dot = block_reduce_sum(local_dot, shared);

    grad_scores4[threadIdx.x] = make_float4(
        p.x * (g.x - dot) * inv_temperature,
        p.y * (g.y - dot) * inv_temperature,
        p.z * (g.z - dot) * inv_temperature,
        p.w * (g.w - dot) * inv_temperature);
}

__global__ void temperature_softmax_backward_generic_kernel(
        const float* __restrict__ probabilities,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_scores,
        int batch,
        int width,
        float inv_temperature) {
    const int row = blockIdx.x;
    if (row >= batch) {
        return;
    }

    __shared__ float shared[8];
    const float* p = probabilities + (size_t)row * width;
    const float* g = grad_output + (size_t)row * width;
    float* dx = grad_scores + (size_t)row * width;

    float local_dot = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        local_dot += p[column] * g[column];
    }
    const float dot = block_reduce_sum(local_dot, shared);

    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        dx[column] = p[column] * (g[column] - dot) * inv_temperature;
    }
}

void check_matrix(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.dim() == 2, name, " must have shape [B, D]");
}

}  // namespace

torch::Tensor temperature_softmax_forward(torch::Tensor scores, double temperature) {
    check_matrix(scores, "scores");
    TORCH_CHECK(std::isfinite(temperature) && temperature > 0.0,
                "temperature must be finite and positive");
    TORCH_CHECK(scores.size(1) > 0, "scores must have a non-empty final dimension");

    const c10::cuda::CUDAGuard device_guard(scores.device());
    scores = scores.contiguous();
    const int batch = static_cast<int>(scores.size(0));
    const int width = static_cast<int>(scores.size(1));
    auto probabilities = torch::empty_like(scores);
    if (batch == 0) {
        return probabilities;
    }

    const float inv_temperature = 1.0f / static_cast<float>(temperature);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (width == 1024) {
        temperature_softmax_forward_1024_kernel<<<batch, kThreads, 0, stream>>>(
            scores.data_ptr<float>(),
            probabilities.data_ptr<float>(),
            batch,
            inv_temperature);
    } else {
        temperature_softmax_forward_generic_kernel<<<batch, kThreads, 0, stream>>>(
            scores.data_ptr<float>(),
            probabilities.data_ptr<float>(),
            batch,
            width,
            inv_temperature);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return probabilities;
}

torch::Tensor temperature_softmax_backward(
        torch::Tensor probabilities,
        torch::Tensor grad_output,
        double inv_temperature) {
    check_matrix(probabilities, "probabilities");
    check_matrix(grad_output, "grad_output");
    TORCH_CHECK(probabilities.sizes() == grad_output.sizes(),
                "probabilities and grad_output must have the same shape");
    TORCH_CHECK(probabilities.device() == grad_output.device(),
                "probabilities and grad_output must be on the same device");
    TORCH_CHECK(std::isfinite(inv_temperature) && inv_temperature > 0.0,
                "inverse temperature must be finite and positive");

    const c10::cuda::CUDAGuard device_guard(probabilities.device());
    probabilities = probabilities.contiguous();
    grad_output = grad_output.contiguous();
    const int batch = static_cast<int>(probabilities.size(0));
    const int width = static_cast<int>(probabilities.size(1));
    auto grad_scores = torch::empty_like(probabilities);
    if (batch == 0) {
        return grad_scores;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (width == 1024) {
        temperature_softmax_backward_1024_kernel<<<batch, kThreads, 0, stream>>>(
            probabilities.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_scores.data_ptr<float>(),
            batch,
            static_cast<float>(inv_temperature));
    } else {
        temperature_softmax_backward_generic_kernel<<<batch, kThreads, 0, stream>>>(
            probabilities.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_scores.data_ptr<float>(),
            batch,
            width,
            static_cast<float>(inv_temperature));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "temperature_softmax_forward",
        &temperature_softmax_forward,
        "Temperature softmax forward (CUDA)");
    module.def(
        "temperature_softmax_backward",
        &temperature_softmax_backward,
        "Temperature softmax backward (CUDA)");
}
