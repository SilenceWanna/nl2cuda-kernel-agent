// 最小冒烟 kernel：向量加法。
// 目的：在 Colab(T4) 上确认 nvcc + torch cpp_extension 编译链路可用，
// 与真正的 RBF kernel 逻辑无关——把"编译环境问题"和"kernel 逻辑问题"分开排查。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vadd_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// a, b: 同形状 1D float32 CUDA 张量。返回 a + b。
torch::Tensor vadd(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a, b must have same shape");

    a = a.contiguous();
    b = b.contiguous();
    auto out = torch::empty_like(a);
    int n = a.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vadd_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vadd", &vadd, "vector add (CUDA smoke test)");
}
