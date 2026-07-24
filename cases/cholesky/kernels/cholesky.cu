#include <torch/extension.h>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int BLOCK_SIZE = 32;
constexpr int DIAG_THREADS = 256;
constexpr int DIAG_WARPS = DIAG_THREADS / 32;
constexpr int ELEMENT_THREADS = 256;

#define CUBLAS_CHECK(expr)                                                        \
    do {                                                                          \
        cublasStatus_t status_ = (expr);                                           \
        TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS,                              \
                    #expr, " failed with cuBLAS status ",                         \
                    static_cast<int>(status_));                                    \
    } while (0)

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void cholesky_diagonal_block_kernel(
    float* __restrict__ matrix,
    int n,
    int offset,
    int block_size) {
    extern __shared__ float tile[];
    const int tid = threadIdx.x;
    const int elements = block_size * block_size;

    for (int index = tid; index < elements; index += blockDim.x) {
        const int row = index / block_size;
        const int col = index - row * block_size;
        tile[index] = row >= col
            ? matrix[static_cast<int64_t>(offset + row) * n + offset + col]
            : 0.0f;
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int column = 0; column < block_size; ++column) {
        if (warp == 0) {
            float sum = 0.0f;
            for (int k = lane; k < column; k += 32) {
                const float value = tile[column * block_size + k];
                sum += value * value;
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                tile[column * block_size + column] = sqrtf(
                    tile[column * block_size + column] - sum);
            }
        }
        __syncthreads();

        const float diagonal = tile[column * block_size + column];
        for (int row = column + 1 + warp; row < block_size; row += DIAG_WARPS) {
            float sum = 0.0f;
            for (int k = lane; k < column; k += 32) {
                sum += tile[row * block_size + k]
                    * tile[column * block_size + k];
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                tile[row * block_size + column] =
                    (tile[row * block_size + column] - sum) / diagonal;
            }
        }
        __syncthreads();
    }

    for (int index = tid; index < elements; index += blockDim.x) {
        const int row = index / block_size;
        const int col = index - row * block_size;
        if (row >= col) {
            matrix[static_cast<int64_t>(offset + row) * n + offset + col] = tile[index];
        }
    }
}

__global__ void zero_strict_upper_kernel(float* matrix, int n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(n) * n;
    if (index < elements) {
        const int row = static_cast<int>(index / n);
        const int col = static_cast<int>(index - static_cast<int64_t>(row) * n);
        if (col > row) {
            matrix[index] = 0.0f;
        }
    }
}

__global__ void phi_lower_kernel(float* matrix, int n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(n) * n;
    if (index < elements) {
        const int row = static_cast<int>(index / n);
        const int col = static_cast<int>(index - static_cast<int64_t>(row) * n);
        if (col > row) {
            matrix[index] = 0.0f;
        } else if (col == row) {
            matrix[index] *= 0.5f;
        }
    }
}

__global__ void symmetrize_kernel(
    const float* __restrict__ matrix,
    float* __restrict__ output,
    int n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(n) * n;
    if (index < elements) {
        const int row = static_cast<int>(index / n);
        const int col = static_cast<int>(index - static_cast<int64_t>(row) * n);
        output[index] = 0.5f * (
            matrix[index] + matrix[static_cast<int64_t>(col) * n + row]);
    }
}

void check_square_float_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == tensor.size(1),
                name, " must have shape [N, N]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.size(0) > 0 && tensor.size(0) <= INT32_MAX,
                name, " has an unsupported matrix order");
}

}  // namespace

torch::Tensor cholesky_forward(torch::Tensor input) {
    check_square_float_cuda(input, "A");
    const int n = static_cast<int>(input.size(0));
    auto factor = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        factor.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<size_t>(n) * n * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        cholesky_diagonal_block_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        BLOCK_SIZE * BLOCK_SIZE * static_cast<int>(sizeof(float))));

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    cublasMath_t previous_math;
    CUBLAS_CHECK(cublasGetMathMode(handle, &previous_math));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    const float one = 1.0f;
    const float minus_one = -1.0f;

    float* data = factor.data_ptr<float>();
    for (int offset = 0; offset < n; offset += BLOCK_SIZE) {
        const int block_size = std::min(BLOCK_SIZE, n - offset);
        const size_t shared_bytes = static_cast<size_t>(block_size) * block_size
            * sizeof(float);
        cholesky_diagonal_block_kernel<<<1, DIAG_THREADS, shared_bytes, stream>>>(
            data,
            n,
            offset,
            block_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const int trailing = n - offset - block_size;
        if (trailing == 0) {
            continue;
        }

        float* diagonal = data + static_cast<int64_t>(offset) * n + offset;
        float* panel = data
            + static_cast<int64_t>(offset + block_size) * n + offset;
        CUBLAS_CHECK(cublasStrsm(
            handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_T,
            CUBLAS_DIAG_NON_UNIT,
            block_size,
            trailing,
            &one,
            diagonal,
            n,
            panel,
            n));

        float* trailing_matrix = data
            + static_cast<int64_t>(offset + block_size) * n
            + offset + block_size;
        // Updating both triangles doubles nominal work but reaches much better
        // throughput than a thin-panel SYRK on Ampere.
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            trailing,
            trailing,
            block_size,
            &minus_one,
            panel,
            n,
            panel,
            n,
            &one,
            trailing_matrix,
            n));
    }
    CUBLAS_CHECK(cublasSetMathMode(handle, previous_math));

    const int64_t elements = static_cast<int64_t>(n) * n;
    zero_strict_upper_kernel<<<
        (elements + ELEMENT_THREADS - 1) / ELEMENT_THREADS,
        ELEMENT_THREADS,
        0,
        stream>>>(data, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return factor;
}

torch::Tensor cholesky_backward(torch::Tensor grad_output, torch::Tensor factor) {
    check_square_float_cuda(grad_output, "grad_output");
    check_square_float_cuda(factor, "L");
    TORCH_CHECK(grad_output.sizes() == factor.sizes(),
                "grad_output and L must have the same shape");
    TORCH_CHECK(grad_output.device() == factor.device(),
                "grad_output and L must be on the same device");

    const int n = static_cast<int>(factor.size(0));
    auto work = torch::empty_like(factor);
    auto grad_input = torch::empty_like(factor);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    cublasMath_t previous_math;
    CUBLAS_CHECK(cublasGetMathMode(handle, &previous_math));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    const float one = 1.0f;

    // Row-major work = L^T @ G. In column-major view this is
    // work^T = G^T @ L, so a right-side triangular multiply is sufficient.
    CUBLAS_CHECK(cublasStrmm(
        handle,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT,
        n,
        n,
        &one,
        factor.data_ptr<float>(),
        n,
        grad_output.data_ptr<float>(),
        n,
        work.data_ptr<float>(),
        n));

    const int64_t elements = static_cast<int64_t>(n) * n;
    phi_lower_kernel<<<
        (elements + ELEMENT_THREADS - 1) / ELEMENT_THREADS,
        ELEMENT_THREADS,
        0,
        stream>>>(work.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Row-major X = L^{-T} P. In column-major view solve
    // X^T = P^T L^{-1} from the right.
    CUBLAS_CHECK(cublasStrsm(
        handle,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT,
        n,
        n,
        &one,
        factor.data_ptr<float>(),
        n,
        work.data_ptr<float>(),
        n));

    // Row-major H = X L^{-1}. In column-major view solve
    // H^T = L^{-T} X^T from the left.
    CUBLAS_CHECK(cublasStrsm(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        n,
        n,
        &one,
        factor.data_ptr<float>(),
        n,
        work.data_ptr<float>(),
        n));
    CUBLAS_CHECK(cublasSetMathMode(handle, previous_math));

    symmetrize_kernel<<<
        (elements + ELEMENT_THREADS - 1) / ELEMENT_THREADS,
        ELEMENT_THREADS,
        0,
        stream>>>(
            work.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("cholesky_forward", &cholesky_forward,
               "Blocked Cholesky forward (CUDA)");
    module.def("cholesky_backward", &cholesky_backward,
               "Blocked Cholesky backward (CUDA)");
}
