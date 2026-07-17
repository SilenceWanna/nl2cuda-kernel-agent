#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>

namespace {

constexpr int kThreads = 256;
constexpr int kForwardThreads = 512;

__device__ __forceinline__ int max_index4(float a, float b, float c, float d) {
    float maximum = a;
    int index = 0;
    if (b > maximum) {
        maximum = b;
        index = 1;
    }
    if (c > maximum) {
        maximum = c;
        index = 2;
    }
    if (d > maximum) {
        index = 3;
    }
    return index;
}

__global__ void maxpool_forward_vec4_kernel(
        const float* __restrict__ X,
        float* __restrict__ Y,
        int H,
        int W,
        int output_h,
        int output_w,
        int plane_count) {
    const int pairs_per_row = output_w / 2;
    for (int plane = blockIdx.x; plane < plane_count; plane += gridDim.x) {
        const int input_plane_offset = plane * H * W;
        const int output_plane_offset = plane * output_h * output_w;
        for (int output_row = blockIdx.y * blockDim.y + threadIdx.y;
             output_row < output_h;
             output_row += blockDim.y * gridDim.y) {
            for (int pair_column = threadIdx.x;
                 pair_column < pairs_per_row;
                 pair_column += blockDim.x) {
                const int input_column = pair_column * 4;
                const int input_offset =
                    input_plane_offset + output_row * 2 * W + input_column;

                const float4 top = *reinterpret_cast<const float4*>(X + input_offset);
                const float4 bottom = *reinterpret_cast<const float4*>(X + input_offset + W);
                float2 result;
                result.x = fmaxf(fmaxf(top.x, top.y), fmaxf(bottom.x, bottom.y));
                result.y = fmaxf(fmaxf(top.z, top.w), fmaxf(bottom.z, bottom.w));

                const int output_offset =
                    output_plane_offset + output_row * output_w + pair_column * 2;
                *reinterpret_cast<float2*>(Y + output_offset) = result;
            }
        }
    }
}

__global__ void maxpool_backward_vec4_kernel(
        const float* __restrict__ X,
        const float* __restrict__ grad_Y,
        float* __restrict__ grad_X,
        int H,
        int W,
        int output_h,
        int output_w) {
    const int pairs_per_row = output_w / 2;
    const int64_t plane = blockIdx.x;
    for (int output_row = blockIdx.y * blockDim.y + threadIdx.y;
         output_row < output_h;
         output_row += blockDim.y * gridDim.y) {
        for (int pair_column = threadIdx.x;
             pair_column < pairs_per_row;
             pair_column += blockDim.x) {
            const int input_column = pair_column * 4;
            const int64_t input_offset =
                plane * static_cast<int64_t>(H) * W +
                static_cast<int64_t>(output_row * 2) * W + input_column;
            const int64_t output_offset =
                plane * static_cast<int64_t>(output_h) * output_w +
                static_cast<int64_t>(output_row) * output_w + pair_column * 2;

            const float4 top = *reinterpret_cast<const float4*>(X + input_offset);
            const float4 bottom = *reinterpret_cast<const float4*>(X + input_offset + W);
            const float2 grad = *reinterpret_cast<const float2*>(grad_Y + output_offset);
            const int first = max_index4(top.x, top.y, bottom.x, bottom.y);
            const int second = max_index4(top.z, top.w, bottom.z, bottom.w);

            float4 grad_top = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 grad_bottom = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (first == 0) grad_top.x = grad.x;
            else if (first == 1) grad_top.y = grad.x;
            else if (first == 2) grad_bottom.x = grad.x;
            else grad_bottom.y = grad.x;
            if (second == 0) grad_top.z = grad.y;
            else if (second == 1) grad_top.w = grad.y;
            else if (second == 2) grad_bottom.z = grad.y;
            else grad_bottom.w = grad.y;

            *reinterpret_cast<float4*>(grad_X + input_offset) = grad_top;
            *reinterpret_cast<float4*>(grad_X + input_offset + W) = grad_bottom;
        }
    }
}

__global__ void maxpool_forward_scalar_kernel(
        const float* __restrict__ X,
        float* __restrict__ Y,
        int64_t output_count,
        int H,
        int W,
        int output_h,
        int output_w) {
    const int64_t outputs_per_plane = static_cast<int64_t>(output_h) * output_w;
    for (int64_t output = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output < output_count;
         output += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t plane = output / outputs_per_plane;
        const int within_plane = static_cast<int>(output - plane * outputs_per_plane);
        const int output_row = within_plane / output_w;
        const int output_column = within_plane - output_row * output_w;
        const int64_t input_offset =
            plane * static_cast<int64_t>(H) * W +
            static_cast<int64_t>(output_row * 2) * W + output_column * 2;
        const float a = X[input_offset];
        const float b = X[input_offset + 1];
        const float c = X[input_offset + W];
        const float d = X[input_offset + W + 1];
        Y[output] = fmaxf(fmaxf(a, b), fmaxf(c, d));
    }
}

__global__ void maxpool_backward_scalar_kernel(
        const float* __restrict__ X,
        const float* __restrict__ grad_Y,
        float* __restrict__ grad_X,
        int64_t output_count,
        int H,
        int W,
        int output_h,
        int output_w) {
    const int64_t outputs_per_plane = static_cast<int64_t>(output_h) * output_w;
    for (int64_t output = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output < output_count;
         output += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t plane = output / outputs_per_plane;
        const int within_plane = static_cast<int>(output - plane * outputs_per_plane);
        const int output_row = within_plane / output_w;
        const int output_column = within_plane - output_row * output_w;
        const int64_t input_offset =
            plane * static_cast<int64_t>(H) * W +
            static_cast<int64_t>(output_row * 2) * W + output_column * 2;
        const float a = X[input_offset];
        const float b = X[input_offset + 1];
        const float c = X[input_offset + W];
        const float d = X[input_offset + W + 1];
        const int index = max_index4(a, b, c, d);
        const float grad = grad_Y[output];
        grad_X[input_offset] = index == 0 ? grad : 0.0f;
        grad_X[input_offset + 1] = index == 1 ? grad : 0.0f;
        grad_X[input_offset + W] = index == 2 ? grad : 0.0f;
        grad_X[input_offset + W + 1] = index == 3 ? grad : 0.0f;
    }
}

int launch_blocks(int64_t work_items) {
    const int64_t blocks = (work_items + kThreads - 1) / kThreads;
    return static_cast<int>(blocks > 65535 ? 65535 : blocks);
}

}  // namespace

torch::Tensor maxpool_forward(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must have dtype float32");
    TORCH_CHECK(X.dim() == 4, "X must have shape [N, C, H, W]");
    TORCH_CHECK(X.size(2) > 0 && X.size(3) > 0 &&
                X.size(2) % 2 == 0 && X.size(3) % 2 == 0,
                "H and W must be positive even integers");

    const c10::cuda::CUDAGuard device_guard(X.device());
    X = X.contiguous();
    const int64_t N = X.size(0);
    const int64_t C = X.size(1);
    const int64_t H64 = X.size(2);
    const int64_t W64 = X.size(3);
    TORCH_CHECK(H64 <= INT_MAX && W64 <= INT_MAX, "H and W must fit in int32");
    const int H = static_cast<int>(H64);
    const int W = static_cast<int>(W64);
    const int output_h = H / 2;
    const int output_w = W / 2;
    auto Y = torch::empty({N, C, output_h, output_w}, X.options());
    const int64_t output_count = N * C * output_h * output_w;
    if (output_count == 0) return Y;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (W % 4 == 0 && X.numel() <= INT_MAX) {
        const int64_t planes = N * C;
        TORCH_CHECK(planes <= INT_MAX, "N*C must fit in the CUDA x grid");
        const int pairs_per_row = output_w / 2;
        const int threads_x = pairs_per_row < 32 ? pairs_per_row : 32;
        const int threads_y = kForwardThreads / threads_x;
        const int blocks_y = (output_h + threads_y - 1) / threads_y;
        dim3 threads(threads_x, threads_y);
        const int forward_blocks = planes < 4096 ? static_cast<int>(planes) : 4096;
        dim3 blocks(static_cast<unsigned>(forward_blocks),
                    static_cast<unsigned>(blocks_y > 65535 ? 65535 : blocks_y));
        maxpool_forward_vec4_kernel<<<blocks, threads, 0, stream>>>(
            X.data_ptr<float>(), Y.data_ptr<float>(),
            H, W, output_h, output_w, static_cast<int>(planes));
    } else {
        maxpool_forward_scalar_kernel<<<launch_blocks(output_count), kThreads, 0, stream>>>(
            X.data_ptr<float>(), Y.data_ptr<float>(), output_count,
            H, W, output_h, output_w);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor maxpool_backward(torch::Tensor X, torch::Tensor grad_Y) {
    TORCH_CHECK(X.is_cuda() && grad_Y.is_cuda(), "X and grad_Y must be CUDA tensors");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                grad_Y.scalar_type() == torch::kFloat32,
                "X and grad_Y must have dtype float32");
    TORCH_CHECK(X.dim() == 4 && grad_Y.dim() == 4,
                "X and grad_Y must be 4D tensors");
    TORCH_CHECK(X.device() == grad_Y.device(), "X and grad_Y must share a device");
    TORCH_CHECK(X.size(2) > 0 && X.size(3) > 0 &&
                X.size(2) % 2 == 0 && X.size(3) % 2 == 0,
                "H and W must be positive even integers");
    TORCH_CHECK(grad_Y.size(0) == X.size(0) && grad_Y.size(1) == X.size(1) &&
                grad_Y.size(2) == X.size(2) / 2 && grad_Y.size(3) == X.size(3) / 2,
                "grad_Y has an invalid shape");

    const c10::cuda::CUDAGuard device_guard(X.device());
    X = X.contiguous();
    grad_Y = grad_Y.contiguous();
    const int64_t N = X.size(0);
    const int64_t C = X.size(1);
    const int64_t H64 = X.size(2);
    const int64_t W64 = X.size(3);
    TORCH_CHECK(H64 <= INT_MAX && W64 <= INT_MAX, "H and W must fit in int32");
    const int H = static_cast<int>(H64);
    const int W = static_cast<int>(W64);
    const int output_h = H / 2;
    const int output_w = W / 2;
    auto grad_X = torch::empty_like(X);
    const int64_t output_count = N * C * output_h * output_w;
    if (output_count == 0) return grad_X;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (W % 4 == 0) {
        const int64_t planes = N * C;
        TORCH_CHECK(planes <= INT_MAX, "N*C must fit in the CUDA x grid");
        const int pairs_per_row = output_w / 2;
        const int threads_x = pairs_per_row < 32 ? pairs_per_row : 32;
        const int threads_y = kThreads / threads_x;
        const int blocks_y = (output_h + threads_y - 1) / threads_y;
        dim3 threads(threads_x, threads_y);
        dim3 blocks(static_cast<unsigned>(planes),
                    static_cast<unsigned>(blocks_y > 65535 ? 65535 : blocks_y));
        maxpool_backward_vec4_kernel<<<blocks, threads, 0, stream>>>(
            X.data_ptr<float>(), grad_Y.data_ptr<float>(), grad_X.data_ptr<float>(),
            H, W, output_h, output_w);
    } else {
        maxpool_backward_scalar_kernel<<<launch_blocks(output_count), kThreads, 0, stream>>>(
            X.data_ptr<float>(), grad_Y.data_ptr<float>(), grad_X.data_ptr<float>(),
            output_count, H, W, output_h, output_w);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("maxpool_forward", &maxpool_forward, "2x2 stride-2 max pooling forward (CUDA)");
    module.def("maxpool_backward", &maxpool_backward, "2x2 stride-2 max pooling backward (CUDA)");
}
