#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

template <bool Backward>
__device__ __forceinline__ float2 rotate_pair(float2 value, float sine, float cosine) {
    float2 result;
    if constexpr (Backward) {
        result.x = value.x * cosine + value.y * sine;
        result.y = -value.x * sine + value.y * cosine;
    } else {
        result.x = value.x * cosine - value.y * sine;
        result.y = value.x * sine + value.y * cosine;
    }
    return result;
}

template <bool Backward>
__global__ void rope_float4_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int64_t vector_count,
        int S,
        int H,
        int D,
        float log_base,
        float frequency_ratio) {
    int64_t vector_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector_index >= vector_count) {
        return;
    }

    const int vectors_per_row = D / 4;
    const int64_t row = vector_index / vectors_per_row;
    const int vector_in_row = static_cast<int>(vector_index - row * vectors_per_row);
    const int pair_index = vector_in_row * 2;
    const int position = static_cast<int>((row / H) % S);

    const float frequency0 = expf(-2.0f * pair_index * log_base / D);
    const float frequency1 = frequency0 * frequency_ratio;
    float sine0, cosine0, sine1, cosine1;
    sincosf(position * frequency0, &sine0, &cosine0);
    sincosf(position * frequency1, &sine1, &cosine1);

    const float4 value = reinterpret_cast<const float4*>(input)[vector_index];
    const float2 first = rotate_pair<Backward>(make_float2(value.x, value.y), sine0, cosine0);
    const float2 second = rotate_pair<Backward>(make_float2(value.z, value.w), sine1, cosine1);
    reinterpret_cast<float4*>(output)[vector_index] =
        make_float4(first.x, first.y, second.x, second.y);
}

template <bool Backward>
__global__ void rope_float2_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int64_t pair_count,
        int S,
        int H,
        int D,
        float log_base) {
    int64_t flat_pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat_pair >= pair_count) {
        return;
    }

    const int pairs_per_row = D / 2;
    const int64_t row = flat_pair / pairs_per_row;
    const int pair_index = static_cast<int>(flat_pair - row * pairs_per_row);
    const int position = static_cast<int>((row / H) % S);
    const float frequency = expf(-2.0f * pair_index * log_base / D);
    float sine, cosine;
    sincosf(position * frequency, &sine, &cosine);

    const float2 value = reinterpret_cast<const float2*>(input)[flat_pair];
    reinterpret_cast<float2*>(output)[flat_pair] =
        rotate_pair<Backward>(value, sine, cosine);
}

void validate_input(const torch::Tensor& input, double base) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must have shape [B, S, H, D]");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 &&
                input.size(2) > 0 && input.size(3) > 0,
                "all input dimensions must be positive");
    TORCH_CHECK(input.size(3) % 2 == 0, "the last dimension D must be even");
    TORCH_CHECK(std::isfinite(base) && base > 0.0, "base must be finite and positive");
}

template <bool Backward>
torch::Tensor launch_rope(torch::Tensor input, double base) {
    validate_input(input, base);
    input = input.contiguous();
    auto output = torch::empty_like(input);

    const int S = static_cast<int>(input.size(1));
    const int H = static_cast<int>(input.size(2));
    const int D = static_cast<int>(input.size(3));
    const float log_base = static_cast<float>(std::log(base));

    if (D % 4 == 0) {
        const int64_t vector_count = input.numel() / 4;
        const int blocks = static_cast<int>((vector_count + kThreads - 1) / kThreads);
        const float frequency_ratio = expf(-2.0f * log_base / D);
        rope_float4_kernel<Backward><<<blocks, kThreads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), vector_count,
            S, H, D, log_base, frequency_ratio);
    } else {
        const int64_t pair_count = input.numel() / 2;
        const int blocks = static_cast<int>((pair_count + kThreads - 1) / kThreads);
        rope_float2_kernel<Backward><<<blocks, kThreads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), pair_count,
            S, H, D, log_base);
    }
    return output;
}

}  // namespace

torch::Tensor rope_forward(torch::Tensor input, double base) {
    return launch_rope<false>(std::move(input), base);
}

torch::Tensor rope_backward(torch::Tensor grad_output, double base) {
    return launch_rope<true>(std::move(grad_output), base);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rope_forward", &rope_forward, "RoPE forward (CUDA)");
    module.def("rope_backward", &rope_backward, "RoPE backward (CUDA)");
}
