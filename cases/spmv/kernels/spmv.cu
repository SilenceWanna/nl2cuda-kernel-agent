#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kFeatureDim = 64;
constexpr int kVectorsPerRow = kFeatureDim / 4;
constexpr int kRowsPerForwardBlock = kThreads / kVectorsPerRow;

__global__ void __launch_bounds__(kThreads)
spmv_forward_d64_kernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const float* __restrict__ vals,
    const float4* __restrict__ X,
    float4* __restrict__ Y,
    int M) {
    const int vector = threadIdx.x & (kVectorsPerRow - 1);
    const int row_in_block = threadIdx.x / kVectorsPerRow;
    const int row = blockIdx.x * kRowsPerForwardBlock + row_in_block;
    if (row >= M) {
        return;
    }

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int64_t begin = row_ptr[row];
    const int64_t end = row_ptr[row + 1];
    for (int64_t p = begin; p < end; ++p) {
        const int64_t column = col_idx[p];
        const float value = vals[p];
        const float4 x = X[column * kVectorsPerRow + vector];
        sum.x = fmaf(value, x.x, sum.x);
        sum.y = fmaf(value, x.y, sum.y);
        sum.z = fmaf(value, x.z, sum.z);
        sum.w = fmaf(value, x.w, sum.w);
    }
    Y[static_cast<int64_t>(row) * kVectorsPerRow + vector] = sum;
}

__global__ void spmv_forward_generic_kernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const float* __restrict__ vals,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int D) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= M) {
        return;
    }

    const int64_t begin = row_ptr[row];
    const int64_t end = row_ptr[row + 1];
    for (int d = lane; d < D; d += kWarpSize) {
        float sum = 0.0f;
        for (int64_t p = begin; p < end; ++p) {
            sum = fmaf(vals[p], X[col_idx[p] * static_cast<int64_t>(D) + d], sum);
        }
        Y[static_cast<int64_t>(row) * D + d] = sum;
    }
}

__global__ void __launch_bounds__(kThreads)
spmv_backward_d64_kernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const float* __restrict__ vals,
    const float* __restrict__ X,
    const float* __restrict__ grad_Y,
    float* __restrict__ grad_vals,
    float* __restrict__ grad_X,
    int M) {
    const int row = blockIdx.x;
    if (row >= M) {
        return;
    }

    __shared__ float grad_row[kFeatureDim];
    if (threadIdx.x < kFeatureDim) {
        grad_row[threadIdx.x] =
            grad_Y[static_cast<int64_t>(row) * kFeatureDim + threadIdx.x];
    }
    __syncthreads();

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int64_t begin = row_ptr[row];
    const int64_t end = row_ptr[row + 1];

    for (int64_t p = begin + warp; p < end; p += kWarpsPerBlock) {
        const int64_t column = col_idx[p];
        const float value = vals[p];
        const int64_t x_offset = column * kFeatureDim;

        const float g0 = grad_row[lane];
        const float g1 = grad_row[lane + kWarpSize];
        const float x0 = X[x_offset + lane];
        const float x1 = X[x_offset + lane + kWarpSize];
        float dot = x0 * g0 + x1 * g1;

        atomicAdd(grad_X + x_offset + lane, value * g0);
        atomicAdd(grad_X + x_offset + lane + kWarpSize, value * g1);

#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            grad_vals[p] = dot;
        }
    }
}

__global__ void spmv_backward_generic_kernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const float* __restrict__ vals,
    const float* __restrict__ X,
    const float* __restrict__ grad_Y,
    float* __restrict__ grad_vals,
    float* __restrict__ grad_X,
    int M,
    int D) {
    const int row = blockIdx.x;
    if (row >= M) {
        return;
    }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int64_t begin = row_ptr[row];
    const int64_t end = row_ptr[row + 1];
    const int64_t grad_y_offset = static_cast<int64_t>(row) * D;

    for (int64_t p = begin + warp; p < end; p += kWarpsPerBlock) {
        const int64_t column = col_idx[p];
        const float value = vals[p];
        const int64_t x_offset = column * static_cast<int64_t>(D);
        float dot = 0.0f;
        for (int d = lane; d < D; d += kWarpSize) {
            const float g = grad_Y[grad_y_offset + d];
            dot += X[x_offset + d] * g;
            atomicAdd(grad_X + x_offset + d, value * g);
        }
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            grad_vals[p] = dot;
        }
    }
}

void validate_structure(
    const torch::Tensor& row_ptr,
    const torch::Tensor& col_idx,
    const torch::Tensor& vals,
    int64_t M) {
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be a CUDA tensor");
    TORCH_CHECK(vals.is_cuda(), "vals must be a CUDA tensor");
    TORCH_CHECK(row_ptr.scalar_type() == torch::kInt64, "row_ptr must be int64");
    TORCH_CHECK(col_idx.scalar_type() == torch::kInt64, "col_idx must be int64");
    TORCH_CHECK(vals.scalar_type() == torch::kFloat32, "vals must be float32");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(col_idx.is_contiguous(), "col_idx must be contiguous");
    TORCH_CHECK(vals.is_contiguous(), "vals must be contiguous");
    TORCH_CHECK(row_ptr.dim() == 1 && row_ptr.size(0) == M + 1,
                "row_ptr must have shape [M + 1]");
    TORCH_CHECK(col_idx.dim() == 1, "col_idx must have shape [nnz]");
    TORCH_CHECK(vals.dim() == 1, "vals must have shape [nnz]");
    TORCH_CHECK(col_idx.numel() == vals.numel(),
                "col_idx and vals must have the same length");
}

void validate_dense(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 2, name, " must be a rank-2 tensor");
}

}  // namespace

torch::Tensor spmv_forward(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor vals,
    torch::Tensor X) {
    validate_dense(X, "X");
    const int64_t M64 = row_ptr.numel() - 1;
    TORCH_CHECK(M64 >= 0 && M64 <= std::numeric_limits<int>::max(),
                "M is too large");
    TORCH_CHECK(X.size(0) > 0 && X.size(1) > 0, "X dimensions must be positive");
    TORCH_CHECK(X.size(1) <= std::numeric_limits<int>::max(), "D is too large");
    validate_structure(row_ptr, col_idx, vals, M64);

    const int M = static_cast<int>(M64);
    const int D = static_cast<int>(X.size(1));
    auto Y = torch::empty({M64, X.size(1)}, X.options());
    if (M == 0) {
        return Y;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (D == kFeatureDim) {
        const int blocks = (M + kRowsPerForwardBlock - 1) / kRowsPerForwardBlock;
        spmv_forward_d64_kernel<<<blocks, kThreads, 0, stream>>>(
            row_ptr.data_ptr<int64_t>(),
            col_idx.data_ptr<int64_t>(),
            vals.data_ptr<float>(),
            reinterpret_cast<const float4*>(X.data_ptr<float>()),
            reinterpret_cast<float4*>(Y.data_ptr<float>()),
            M);
    } else {
        const int blocks = (M + kWarpsPerBlock - 1) / kWarpsPerBlock;
        spmv_forward_generic_kernel<<<blocks, kThreads, 0, stream>>>(
            row_ptr.data_ptr<int64_t>(),
            col_idx.data_ptr<int64_t>(),
            vals.data_ptr<float>(),
            X.data_ptr<float>(),
            Y.data_ptr<float>(),
            M,
            D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

std::vector<torch::Tensor> spmv_backward(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor vals,
    torch::Tensor X,
    torch::Tensor grad_Y) {
    validate_dense(X, "X");
    validate_dense(grad_Y, "grad_Y");
    const int64_t M64 = row_ptr.numel() - 1;
    TORCH_CHECK(M64 >= 0 && M64 <= std::numeric_limits<int>::max(),
                "M is too large");
    TORCH_CHECK(X.size(0) > 0 && X.size(1) > 0, "X dimensions must be positive");
    TORCH_CHECK(X.size(1) <= std::numeric_limits<int>::max(), "D is too large");
    TORCH_CHECK(grad_Y.size(0) == M64 && grad_Y.size(1) == X.size(1),
                "grad_Y must have shape [M, D]");
    validate_structure(row_ptr, col_idx, vals, M64);

    auto grad_vals = torch::empty_like(vals);
    auto grad_X = torch::empty_like(X);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        grad_X.data_ptr<float>(), 0, grad_X.numel() * sizeof(float), stream));

    const int M = static_cast<int>(M64);
    const int D = static_cast<int>(X.size(1));
    if (M > 0 && vals.numel() > 0) {
        if (D == kFeatureDim) {
            spmv_backward_d64_kernel<<<M, kThreads, 0, stream>>>(
                row_ptr.data_ptr<int64_t>(),
                col_idx.data_ptr<int64_t>(),
                vals.data_ptr<float>(),
                X.data_ptr<float>(),
                grad_Y.data_ptr<float>(),
                grad_vals.data_ptr<float>(),
                grad_X.data_ptr<float>(),
                M);
        } else {
            spmv_backward_generic_kernel<<<M, kThreads, 0, stream>>>(
                row_ptr.data_ptr<int64_t>(),
                col_idx.data_ptr<int64_t>(),
                vals.data_ptr<float>(),
                X.data_ptr<float>(),
                grad_Y.data_ptr<float>(),
                grad_vals.data_ptr<float>(),
                grad_X.data_ptr<float>(),
                M,
                D);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {grad_vals, grad_X};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("spmv_forward", &spmv_forward, "CSR SpMM/SpMV forward (CUDA)");
    module.def("spmv_backward", &spmv_backward, "CSR SpMM/SpMV backward (CUDA)");
}
