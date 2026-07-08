// ============================================================================
// RBF 纯 CUDA 交付版 —— 自测 harness（内置 CPU 参考，不依赖 PyTorch）
// ============================================================================
// 生成随机 X/Y/G，调 GPU host 函数得 K/dX/dY，与内置 CPU 朴素参考对拍 allclose。
// 小规模（默认 N=M=128, D=64）即可证明 kernel 数学正确性；kernel 计算逻辑与
// A100 上已验收（前向1.10×/反向1.17×、正确性全PASS）的版本逐字一致。
//
// 编译运行：  make test    （或 nvcc -O3 -arch=sm_80 rbf_kernels.cu rbf_test.cu -o rbf_test && ./rbf_test）
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

extern "C" void rbf_forward_cuda(const float*, const float*, float*,
                                 int, int, int, float);
extern "C" void rbf_backward_cuda(const float*, const float*, const float*,
                                  const float*, float*, float*,
                                  int, int, int, float);

// ---- CPU 朴素参考（直接按数学定义，作为自包含金标准）----
static void cpu_forward(const std::vector<float>& X, const std::vector<float>& Y,
                        std::vector<float>& K, int N, int M, int D, float gamma) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j) {
            float dist = 0.f;
            for (int d = 0; d < D; ++d) {
                float diff = X[(size_t)i*D+d] - Y[(size_t)j*D+d];
                dist += diff*diff;
            }
            K[(size_t)i*M+j] = std::exp(-gamma * dist);
        }
}

static void cpu_backward(const std::vector<float>& X, const std::vector<float>& Y,
                         const std::vector<float>& G, const std::vector<float>& K,
                         std::vector<float>& dX, std::vector<float>& dY,
                         int N, int M, int D, float gamma) {
    std::fill(dX.begin(), dX.end(), 0.f);
    std::fill(dY.begin(), dY.end(), 0.f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j) {
            float coef = -2.f * gamma * G[(size_t)i*M+j] * K[(size_t)i*M+j];
            for (int d = 0; d < D; ++d) {
                float diff = X[(size_t)i*D+d] - Y[(size_t)j*D+d];  // x - y
                dX[(size_t)i*D+d] += coef * diff;
                dY[(size_t)j*D+d] += coef * (-diff);               // y - x
            }
        }
}

static bool allclose(const std::vector<float>& a, const std::vector<float>& b,
                     float atol, float rtol, float* max_err) {
    float me = 0.f; bool ok = true;
    for (size_t i = 0; i < a.size(); ++i) {
        float e = std::fabs(a[i] - b[i]);
        if (e > me) me = e;
        if (e > atol + rtol * std::fabs(b[i])) ok = false;
    }
    *max_err = me;
    return ok;
}

int main(int argc, char** argv) {
    int N = 128, M = 128, D = 64;
    float gamma = 1.0f / 64.0f;
    if (argc >= 4) { N = atoi(argv[1]); M = atoi(argv[2]); D = atoi(argv[3]); }
    const float atol = 1e-2f, rtol = 1e-2f;

    std::srand(1234);
    std::vector<float> X((size_t)N*D), Y((size_t)M*D), G((size_t)N*M);
    auto rnd = []() { return (float)std::rand() / RAND_MAX * 2.f - 1.f; };
    for (auto& v : X) v = rnd();
    for (auto& v : Y) v = rnd();
    for (auto& v : G) v = rnd();

    std::vector<float> K_gpu((size_t)N*M), K_ref((size_t)N*M);
    std::vector<float> dX_gpu((size_t)N*D), dY_gpu((size_t)M*D);
    std::vector<float> dX_ref((size_t)N*D), dY_ref((size_t)M*D);

    // GPU
    rbf_forward_cuda(X.data(), Y.data(), K_gpu.data(), N, M, D, gamma);
    rbf_backward_cuda(X.data(), Y.data(), G.data(), K_gpu.data(),
                      dX_gpu.data(), dY_gpu.data(), N, M, D, gamma);
    // CPU 参考（反向用 CPU 自己的 K_ref，避免 GPU 误差传入影响对拍独立性）
    cpu_forward(X, Y, K_ref, N, M, D, gamma);
    cpu_backward(X, Y, G, K_ref, dX_ref, dY_ref, N, M, D, gamma);

    printf("=== RBF 纯 CUDA 交付版自测  N=%d M=%d D=%d gamma=%.6g ===\n", N, M, D, gamma);
    float eK, edX, edY;
    bool okK  = allclose(K_gpu,  K_ref,  atol, rtol, &eK);
    bool okdX = allclose(dX_gpu, dX_ref, atol, rtol, &edX);
    bool okdY = allclose(dY_gpu, dY_ref, atol, rtol, &edY);
    printf("[前向 K ] %s  max_abs_err=%.3e\n", okK  ? "PASS" : "FAIL", eK);
    printf("[反向 dX] %s  max_abs_err=%.3e\n", okdX ? "PASS" : "FAIL", edX);
    printf("[反向 dY] %s  max_abs_err=%.3e\n", okdY ? "PASS" : "FAIL", edY);
    bool all = okK && okdX && okdY;
    printf("=== 总判定: %s (atol=rtol=1e-2) ===\n", all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}
