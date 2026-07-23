// Unordered segment softmax for float32 values.
//
// Values are viewed as [N, F].  Segment ids select the reduction group for a
// row, while each feature f is normalized independently.  The forward pass is
// numerically stable: atomic segment maxima, atomic sums of exp(x - max), then
// a normalization pass.  Backward reuses the saved probability tensor:
// dx = p * (g - segment_sum(g * p)).

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

__device__ __forceinline__ unsigned int float_to_ordered_key(float value) {
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float ordered_key_to_float(unsigned int key) {
    const unsigned int bits =
        (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key;
    return __uint_as_float(bits);
}

__global__ void initialize_forward_stats_kernel(
    unsigned int* __restrict__ maximum_keys,
    float* __restrict__ sums,
    int64_t count) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = start; index < count; index += stride) {
        maximum_keys[index] = 0x007fffffu;
        sums[index] = 0.0f;
    }
}

__global__ void segment_max_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ segment_ids,
    unsigned int* __restrict__ maximum_keys,
    int64_t rows,
    int64_t features,
    int64_t num_segments) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = start; row < rows; row += stride) {
        const int64_t segment = segment_ids[row];
        if (segment < 0 || segment >= num_segments) {
            continue;
        }
        const int64_t value_offset = row * features;
        const int64_t stat_offset = segment * features;
        for (int64_t feature = 0; feature < features; ++feature) {
            atomicMax(
                maximum_keys + stat_offset + feature,
                float_to_ordered_key(values[value_offset + feature]));
        }
    }
}

__global__ void segment_exp_sum_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ segment_ids,
    const unsigned int* __restrict__ maximum_keys,
    float* __restrict__ sums,
    float* __restrict__ exponentials,
    int64_t rows,
    int64_t features,
    int64_t num_segments) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = start; row < rows; row += stride) {
        const int64_t segment = segment_ids[row];
        if (segment < 0 || segment >= num_segments) {
            continue;
        }
        const int64_t value_offset = row * features;
        const int64_t stat_offset = segment * features;
        for (int64_t feature = 0; feature < features; ++feature) {
            const float exponent = expf(
                values[value_offset + feature] -
                ordered_key_to_float(maximum_keys[stat_offset + feature]));
            exponentials[value_offset + feature] = exponent;
            atomicAdd(sums + stat_offset + feature, exponent);
        }
    }
}

__global__ void segment_normalize_kernel(
    float* __restrict__ output,
    const int64_t* __restrict__ segment_ids,
    const float* __restrict__ sums,
    int64_t rows,
    int64_t features,
    int64_t num_segments) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = start; row < rows; row += stride) {
        const int64_t segment = segment_ids[row];
        const int64_t value_offset = row * features;
        if (segment < 0 || segment >= num_segments) {
            for (int64_t feature = 0; feature < features; ++feature) {
                output[value_offset + feature] = 0.0f;
            }
            continue;
        }
        const int64_t stat_offset = segment * features;
        for (int64_t feature = 0; feature < features; ++feature) {
            output[value_offset + feature] /= sums[stat_offset + feature];
        }
    }
}

template <int Features>
__global__ void segment_max_fixed_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ segment_ids,
    unsigned int* __restrict__ maximum_keys,
    int64_t elements,
    int64_t num_segments) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = start; index < elements; index += stride) {
        const int64_t row = index / Features;
        const int feature = static_cast<int>(index - row * Features);
        const int64_t segment = segment_ids[row];
        if (segment >= 0 && segment < num_segments) {
            atomicMax(
                maximum_keys + segment * Features + feature,
                float_to_ordered_key(values[index]));
        }
    }
}

template <int Features>
__global__ void segment_exp_sum_fixed_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ segment_ids,
    const unsigned int* __restrict__ maximum_keys,
    float* __restrict__ sums,
    float* __restrict__ exponentials,
    int64_t elements,
    int64_t num_segments) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = start; index < elements; index += stride) {
        const int64_t row = index / Features;
        const int feature = static_cast<int>(index - row * Features);
        const int64_t segment = segment_ids[row];
        if (segment < 0 || segment >= num_segments) {
            continue;
        }
        const int64_t stat_index = segment * Features + feature;
        const float exponent = expf(
            values[index] - ordered_key_to_float(maximum_keys[stat_index]));
        exponentials[index] = exponent;
        atomicAdd(sums + stat_index, exponent);
    }
}

template <int Features>
__global__ void segment_normalize_fixed_kernel(
    float* __restrict__ output,
    const int64_t* __restrict__ segment_ids,
    const float* __restrict__ sums,
    int64_t elements,
    int64_t num_segments) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = start; index < elements; index += stride) {
        const int64_t row = index / Features;
        const int feature = static_cast<int>(index - row * Features);
        const int64_t segment = segment_ids[row];
        if (segment >= 0 && segment < num_segments) {
            output[index] /= sums[segment * Features + feature];
        } else {
            output[index] = 0.0f;
        }
    }
}

__global__ void segment_dot_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ segment_ids,
    float* __restrict__ dots,
    int64_t rows,
    int64_t features,
    int64_t num_segments) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = start; row < rows; row += stride) {
        const int64_t segment = segment_ids[row];
        if (segment < 0 || segment >= num_segments) {
            continue;
        }
        const int64_t value_offset = row * features;
        const int64_t stat_offset = segment * features;
        for (int64_t feature = 0; feature < features; ++feature) {
            atomicAdd(
                dots + stat_offset + feature,
                probabilities[value_offset + feature] * grad_output[value_offset + feature]);
        }
    }
}

__global__ void segment_backward_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ segment_ids,
    const float* __restrict__ dots,
    float* __restrict__ grad_values,
    int64_t rows,
    int64_t features,
    int64_t num_segments) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = start; row < rows; row += stride) {
        const int64_t segment = segment_ids[row];
        const int64_t value_offset = row * features;
        if (segment < 0 || segment >= num_segments) {
            for (int64_t feature = 0; feature < features; ++feature) {
                grad_values[value_offset + feature] = 0.0f;
            }
            continue;
        }
        const int64_t stat_offset = segment * features;
        for (int64_t feature = 0; feature < features; ++feature) {
            const float probability = probabilities[value_offset + feature];
            grad_values[value_offset + feature] = probability * (
                grad_output[value_offset + feature] - dots[stat_offset + feature]);
        }
    }
}

void check_values(const torch::Tensor& values, const char* name) {
    TORCH_CHECK(values.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(values.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(values.dim() >= 1, name, " must have shape [N, ...]");
}

void check_segment_ids(const torch::Tensor& segment_ids, int64_t rows) {
    TORCH_CHECK(segment_ids.is_cuda(), "segment_ids must be a CUDA tensor");
    TORCH_CHECK(segment_ids.scalar_type() == torch::kInt64,
                "segment_ids must be int64");
    TORCH_CHECK(segment_ids.is_contiguous(), "segment_ids must be contiguous");
    TORCH_CHECK(segment_ids.dim() == 1, "segment_ids must have shape [N]");
    TORCH_CHECK(segment_ids.size(0) == rows,
                "segment_ids length must equal values.size(0)");
}

int64_t feature_count(const torch::Tensor& values) {
    int64_t features = 1;
    for (int64_t dim = 1; dim < values.dim(); ++dim) {
        features *= values.size(dim);
    }
    return features;
}

int launch_blocks(int64_t count) {
    const int64_t required = (count + kThreads - 1) / kThreads;
    return static_cast<int>(std::min<int64_t>(std::max<int64_t>(required, 1), kMaxBlocks));
}

template <int Features>
void launch_forward_fixed(
    const torch::Tensor& values,
    const torch::Tensor& segment_ids,
    unsigned int* maximum_keys,
    float* sums,
    torch::Tensor& output,
    int64_t rows,
    int64_t num_segments,
    cudaStream_t stream) {
    const int64_t elements = rows * Features;
    const int blocks = launch_blocks(elements);
    segment_max_fixed_kernel<Features><<<blocks, kThreads, 0, stream>>>(
        values.data_ptr<float>(), segment_ids.data_ptr<int64_t>(), maximum_keys,
        elements, num_segments);
    segment_exp_sum_fixed_kernel<Features><<<blocks, kThreads, 0, stream>>>(
        values.data_ptr<float>(), segment_ids.data_ptr<int64_t>(), maximum_keys,
        sums, output.data_ptr<float>(), elements, num_segments);
    segment_normalize_fixed_kernel<Features><<<blocks, kThreads, 0, stream>>>(
        output.data_ptr<float>(), segment_ids.data_ptr<int64_t>(), sums,
        elements, num_segments);
}

}  // namespace

torch::Tensor segment_softmax_forward(
    torch::Tensor values,
    torch::Tensor segment_ids,
    int64_t num_segments) {
    check_values(values, "values");
    TORCH_CHECK(num_segments > 0, "num_segments must be positive");
    check_segment_ids(segment_ids, values.size(0));
    TORCH_CHECK(values.device() == segment_ids.device(),
                "values and segment_ids must be on the same device");

    const c10::cuda::CUDAGuard device_guard(values.device());
    const int64_t rows = values.size(0);
    const int64_t features = feature_count(values);
    auto output = torch::empty_like(values);
    if (rows == 0 || features == 0) {
        return output;
    }

    const int64_t stat_count = num_segments * features;
    auto stats = torch::empty({2, num_segments, features}, values.options());
    float* stats_data = stats.data_ptr<float>();
    auto* maximum_keys = reinterpret_cast<unsigned int*>(stats_data);
    float* sums = stats_data + stat_count;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    initialize_forward_stats_kernel<<<launch_blocks(stat_count), kThreads, 0, stream>>>(
        maximum_keys, sums, stat_count);
    if (features == 1) {
        launch_forward_fixed<1>(
            values, segment_ids, maximum_keys, sums, output,
            rows, num_segments, stream);
    } else if (features == 4) {
        launch_forward_fixed<4>(
            values, segment_ids, maximum_keys, sums, output,
            rows, num_segments, stream);
    } else if (features == 8) {
        launch_forward_fixed<8>(
            values, segment_ids, maximum_keys, sums, output,
            rows, num_segments, stream);
    } else if (features == 16) {
        launch_forward_fixed<16>(
            values, segment_ids, maximum_keys, sums, output,
            rows, num_segments, stream);
    } else {
        segment_max_kernel<<<launch_blocks(rows), kThreads, 0, stream>>>(
            values.data_ptr<float>(), segment_ids.data_ptr<int64_t>(), maximum_keys,
            rows, features, num_segments);
        segment_exp_sum_kernel<<<launch_blocks(rows), kThreads, 0, stream>>>(
            values.data_ptr<float>(), segment_ids.data_ptr<int64_t>(), maximum_keys, sums,
            output.data_ptr<float>(), rows, features, num_segments);
        segment_normalize_kernel<<<launch_blocks(rows), kThreads, 0, stream>>>(
            output.data_ptr<float>(), segment_ids.data_ptr<int64_t>(), sums,
            rows, features, num_segments);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor segment_softmax_backward(
    torch::Tensor probabilities,
    torch::Tensor grad_output,
    torch::Tensor segment_ids,
    int64_t num_segments) {
    check_values(probabilities, "probabilities");
    check_values(grad_output, "grad_output");
    TORCH_CHECK(probabilities.sizes() == grad_output.sizes(),
                "probabilities and grad_output must have the same shape");
    TORCH_CHECK(num_segments > 0, "num_segments must be positive");
    check_segment_ids(segment_ids, probabilities.size(0));
    TORCH_CHECK(probabilities.device() == grad_output.device() &&
                probabilities.device() == segment_ids.device(),
                "all tensors must be on the same device");

    const c10::cuda::CUDAGuard device_guard(probabilities.device());
    const int64_t rows = probabilities.size(0);
    const int64_t features = feature_count(probabilities);
    auto grad_values = torch::empty_like(probabilities);
    if (rows == 0 || features == 0) {
        return grad_values;
    }

    const int64_t stat_count = num_segments * features;
    auto dots = torch::empty({num_segments, features}, probabilities.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        dots.data_ptr<float>(), 0, static_cast<size_t>(stat_count) * sizeof(float), stream));

    segment_dot_kernel<<<launch_blocks(rows), kThreads, 0, stream>>>(
        probabilities.data_ptr<float>(), grad_output.data_ptr<float>(),
        segment_ids.data_ptr<int64_t>(), dots.data_ptr<float>(),
        rows, features, num_segments);
    segment_backward_kernel<<<launch_blocks(rows), kThreads, 0, stream>>>(
        probabilities.data_ptr<float>(), grad_output.data_ptr<float>(),
        segment_ids.data_ptr<int64_t>(), dots.data_ptr<float>(),
        grad_values.data_ptr<float>(), rows, features, num_segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_values;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "segment_softmax_forward",
        &segment_softmax_forward,
        "Segment softmax forward (CUDA)");
    module.def(
        "segment_softmax_backward",
        &segment_softmax_backward,
        "Segment softmax backward (CUDA)");
}
