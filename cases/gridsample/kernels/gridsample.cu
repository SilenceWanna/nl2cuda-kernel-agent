#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

struct BilinearCoordinates {
    int x0;
    int x1;
    int y0;
    int y1;
    bool x0_valid;
    bool x1_valid;
    bool y0_valid;
    bool y1_valid;
    float wx0;
    float wx1;
    float wy0;
    float wy1;
};

__device__ __forceinline__ float pixel_coordinate(
        float coordinate, int size) {
    if (isnan(coordinate)) {
        coordinate = -1.0f;
    }
    return ((coordinate + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
}

__device__ __forceinline__ BilinearCoordinates make_coordinates(
        float gx, float gy, int H, int W) {
    const float ix = pixel_coordinate(gx, W);
    const float iy = pixel_coordinate(gy, H);
    const int x0 = static_cast<int>(floorf(ix));
    const int y0 = static_cast<int>(floorf(iy));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    BilinearCoordinates p;
    p.x0 = min(max(x0, 0), W - 1);
    p.x1 = min(max(x1, 0), W - 1);
    p.y0 = min(max(y0, 0), H - 1);
    p.y1 = min(max(y1, 0), H - 1);
    p.x0_valid = x0 >= 0 && x0 < W;
    p.x1_valid = x1 >= 0 && x1 < W;
    p.y0_valid = y0 >= 0 && y0 < H;
    p.y1_valid = y1 >= 0 && y1 < H;
    p.wx1 = ix - static_cast<float>(x0);
    p.wy1 = iy - static_cast<float>(y0);
    p.wx0 = 1.0f - p.wx1;
    p.wy0 = 1.0f - p.wy1;
    return p;
}

__device__ __forceinline__ float load_value(
        const float* __restrict__ X, int64_t plane, int H, int W,
        int y, int x, bool valid) {
    return valid ? X[plane * static_cast<int64_t>(H) * W +
                      static_cast<int64_t>(y) * W + x] : 0.0f;
}

__global__ void gridsample_forward_kernel(
        const float* __restrict__ X,
        const float* __restrict__ grid,
        float* __restrict__ Y,
        int64_t spatial_count,
        int64_t C,
        int H,
        int W,
        int OH,
        int OW) {
    for (int64_t spatial = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         spatial < spatial_count;
         spatial += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t n = spatial / (static_cast<int64_t>(OH) * OW);
        const int64_t within = spatial - n * static_cast<int64_t>(OH) * OW;
        const int oh = static_cast<int>(within / OW);
        const int ow = static_cast<int>(within - static_cast<int64_t>(oh) * OW);
        const int64_t grid_offset = spatial * 2;
        const BilinearCoordinates p = make_coordinates(
            grid[grid_offset], grid[grid_offset + 1], H, W);
        const float w00 = p.wx0 * p.wy0;
        const float w01 = p.wx1 * p.wy0;
        const float w10 = p.wx0 * p.wy1;
        const float w11 = p.wx1 * p.wy1;
        for (int64_t c = 0; c < C; ++c) {
            const int64_t plane = n * C + c;
            const float v00 = load_value(X, plane, H, W, p.y0, p.x0,
                                         p.x0_valid && p.y0_valid);
            const float v01 = load_value(X, plane, H, W, p.y0, p.x1,
                                         p.x1_valid && p.y0_valid);
            const float v10 = load_value(X, plane, H, W, p.y1, p.x0,
                                         p.x0_valid && p.y1_valid);
            const float v11 = load_value(X, plane, H, W, p.y1, p.x1,
                                         p.x1_valid && p.y1_valid);
            const int64_t output_offset =
                (plane * OH + oh) * static_cast<int64_t>(OW) + ow;
            Y[output_offset] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        }
    }
}

__global__ void gridsample_backward_kernel(
        const float* __restrict__ grid,
        const float* __restrict__ grad_Y,
        float* __restrict__ grad_X,
        int64_t spatial_count,
        int64_t C,
        int H,
        int W,
        int OH,
        int OW) {
    for (int64_t spatial = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         spatial < spatial_count;
         spatial += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t n = spatial / (static_cast<int64_t>(OH) * OW);
        const int64_t within = spatial - n * static_cast<int64_t>(OH) * OW;
        const int oh = static_cast<int>(within / OW);
        const int ow = static_cast<int>(within - static_cast<int64_t>(oh) * OW);
        const int64_t grid_offset = spatial * 2;
        const BilinearCoordinates p = make_coordinates(
            grid[grid_offset], grid[grid_offset + 1], H, W);
        const float w00 = p.wx0 * p.wy0;
        const float w01 = p.wx1 * p.wy0;
        const float w10 = p.wx0 * p.wy1;
        const float w11 = p.wx1 * p.wy1;
        for (int64_t c = 0; c < C; ++c) {
            const int64_t plane = n * C + c;
            const int64_t output_offset =
                (plane * OH + oh) * static_cast<int64_t>(OW) + ow;
            const float g = grad_Y[output_offset];
            const bool v00_valid = p.x0_valid && p.y0_valid;
            const bool v01_valid = p.x1_valid && p.y0_valid;
            const bool v10_valid = p.x0_valid && p.y1_valid;
            const bool v11_valid = p.x1_valid && p.y1_valid;

            if (g != 0.0f) {
                const int64_t base = plane * static_cast<int64_t>(H) * W;
                if (v00_valid) atomicAdd(grad_X + base +
                    static_cast<int64_t>(p.y0) * W + p.x0, g * w00);
                if (v01_valid) atomicAdd(grad_X + base +
                    static_cast<int64_t>(p.y0) * W + p.x1, g * w01);
                if (v10_valid) atomicAdd(grad_X + base +
                    static_cast<int64_t>(p.y1) * W + p.x0, g * w10);
                if (v11_valid) atomicAdd(grad_X + base +
                    static_cast<int64_t>(p.y1) * W + p.x1, g * w11);
            }
        }
    }
}

int launch_blocks(int64_t count) {
    const int64_t blocks = (count + kThreads - 1) / kThreads;
    return static_cast<int>(std::min<int64_t>(blocks, 65535));
}

void check_common(torch::Tensor X, torch::Tensor grid) {
    TORCH_CHECK(X.is_cuda() && grid.is_cuda(), "X and grid must be CUDA tensors");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                grid.scalar_type() == torch::kFloat32,
                "X and grid must have dtype float32");
    TORCH_CHECK(X.dim() == 4 && grid.dim() == 4,
                "X must be [N,C,H,W] and grid must be [N,OH,OW,2]");
    TORCH_CHECK(X.device() == grid.device(), "X and grid must share a device");
    TORCH_CHECK(grid.size(0) == X.size(0) && grid.size(3) == 2,
                "grid has an invalid shape");
    TORCH_CHECK(X.size(2) > 0 && X.size(3) > 0 &&
                grid.size(1) > 0 && grid.size(2) > 0,
                "all spatial dimensions must be positive");
    TORCH_CHECK(X.size(2) <= INT_MAX && X.size(3) <= INT_MAX &&
                grid.size(1) <= INT_MAX && grid.size(2) <= INT_MAX,
                "spatial dimensions must fit in int32");
}

}  // namespace

torch::Tensor gridsample_forward(torch::Tensor X, torch::Tensor grid) {
    check_common(X, grid);
    const c10::cuda::CUDAGuard device_guard(X.device());
    X = X.contiguous();
    grid = grid.contiguous();
    const int64_t N = X.size(0);
    const int64_t C = X.size(1);
    const int H = static_cast<int>(X.size(2));
    const int W = static_cast<int>(X.size(3));
    const int OH = static_cast<int>(grid.size(1));
    const int OW = static_cast<int>(grid.size(2));
    auto Y = torch::empty({N, C, OH, OW}, X.options());
    const int64_t spatial_count = N * static_cast<int64_t>(OH) * OW;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gridsample_forward_kernel<<<launch_blocks(spatial_count), kThreads, 0, stream>>>(
        X.data_ptr<float>(), grid.data_ptr<float>(), Y.data_ptr<float>(),
        spatial_count, C, H, W, OH, OW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor gridsample_backward(
        torch::Tensor grid, torch::Tensor grad_Y,
        int64_t input_h, int64_t input_w) {
    TORCH_CHECK(grid.is_cuda(), "grid must be a CUDA tensor");
    TORCH_CHECK(grid.scalar_type() == torch::kFloat32,
                "grid must have dtype float32");
    TORCH_CHECK(grid.dim() == 4 && grid.size(3) == 2,
                "grid must have shape [N,OH,OW,2]");
    TORCH_CHECK(grad_Y.is_cuda() && grad_Y.scalar_type() == torch::kFloat32 &&
                grad_Y.dim() == 4, "grad_Y must be a CUDA float32 4D tensor");
    TORCH_CHECK(grad_Y.device() == grid.device(),
                "grid and grad_Y must share a device");
    TORCH_CHECK(grad_Y.size(0) == grid.size(0) &&
                grad_Y.size(2) == grid.size(1) && grad_Y.size(3) == grid.size(2),
                "grad_Y has an invalid shape");
    TORCH_CHECK(input_h > 0 && input_h <= INT_MAX &&
                input_w > 0 && input_w <= INT_MAX,
                "input H and W must be positive int32 values");
    const c10::cuda::CUDAGuard device_guard(grid.device());
    grid = grid.contiguous();
    grad_Y = grad_Y.contiguous();
    const int64_t N = grid.size(0);
    const int64_t C = grad_Y.size(1);
    const int H = static_cast<int>(input_h);
    const int W = static_cast<int>(input_w);
    const int OH = static_cast<int>(grid.size(1));
    const int OW = static_cast<int>(grid.size(2));
    auto grad_X = torch::zeros({N, C, H, W}, grad_Y.options());
    const int64_t spatial_count = N * static_cast<int64_t>(OH) * OW;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gridsample_backward_kernel<<<launch_blocks(spatial_count), kThreads, 0, stream>>>(
        grid.data_ptr<float>(), grad_Y.data_ptr<float>(), grad_X.data_ptr<float>(),
        spatial_count, C, H, W, OH, OW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("gridsample_forward", &gridsample_forward,
               "Bilinear grid sample forward (CUDA)");
    module.def("gridsample_backward", &gridsample_backward,
               "Bilinear grid sample backward (CUDA)");
}
