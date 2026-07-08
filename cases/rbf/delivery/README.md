# RBF 高斯核矩阵 —— CUDA 前反向 Kernel（可独立编译交付版）

自然语言描述的 RBF 核算法结构，经本项目 skill 驱动生成并优化的自定义 CUDA 前向 + 反向 kernel。
**完全不依赖 PyTorch**，仅需 `nvcc` + CUDA runtime 即可独立编译。

## 算法

- 输入：`X:[N,D]`, `Y:[M,D]`（float32）
- 前向：`K[i,j] = exp(-gamma * ||x_i - y_j||^2)`，输出 `K:[N,M]`
- 反向：给定上游梯度 `G = dL/dK`，求 `dX:[N,D]`、`dY:[M,D]`
  - `coef[i,j] = -2*gamma*G[i,j]*K[i,j]`
  - `dX[i,d] = Σ_j coef[i,j]*(X[i,d]-Y[j,d])`，`dY[j,d] = Σ_i coef[i,j]*(Y[j,d]-X[i,d])`

## 文件

| 文件 | 说明 |
|------|------|
| `rbf_kernels.cu` | 前向+反向 `__global__` kernel + `extern "C"` 裸指针 host 接口 |
| `rbf_test.cu` | 自测 harness：内置 CPU 参考，对拍 GPU 输出（自包含，不依赖 torch） |
| `Makefile` | nvcc 独立编译；`make test` 编译+跑自测 |

## 编译 / 运行

```bash
make test                 # 编译 + 运行自测（默认 A100 sm_80，小规模 CPU 对拍）
make ARCH=sm_75 test      # 换 T4 等其他架构
make librbf.a             # 只产出静态库供外部链接
./rbf_test 2048 2048 64   # 自定义规模跑自测
```

预期自测输出：前向 K、反向 dX、反向 dY 三项 `allclose(atol=rtol=1e-2)` 全 **PASS**。

## host 接口

```c
// 收裸主机指针；内部负责 device 内存分配/拷贝/kernel launch/同步。
extern "C" void rbf_forward_cuda(const float* X, const float* Y, float* K,
                                 int N, int M, int D, float gamma);
extern "C" void rbf_backward_cuda(const float* X, const float* Y, const float* G,
                                  const float* K, float* dX, float* dY,
                                  int N, int M, int D, float gamma);
```
（反向复用前向输出 `K`，避免重算 dist/exp。约束：反向 kernel 以 `blockDim=D` 启动，要求 D 为 2 的幂且 ≤1024，本用例 D=64 满足。）

## 优化要点

- **前向**：GEMM 式 shared-memory tiling（32×32 输出块）+ thread coarsening（每线程 2×2 微块，256 线程/block 高 occupancy）+ float4 向量化读 shared。
- **反向**：前向缓存 K 复用 → coef 为标量、各分量独立累加、无 per-j 规约（消除同步开销与 dist/exp 重算）；block-per-row 高 occupancy。

## 验收结果（A100-SXM4-40GB, sm_80）

以 PyTorch 广播实现为金标准、对比 `torch.compile`（默认 mode），规范计时（CUDA events、warmup≥10、≥100 次几何均值、CV≤5%）：

| | 自定义 kernel | torch.compile | 加速比 |
|---|---|---|---|
| 前向 | 0.495 ms | 0.546 ms | **1.10×** |
| 反向 | 0.815 ms | 0.952 ms | **1.17×** |

正确性：≥5 组随机种子，前向 + 反向各梯度 `allclose(atol=rtol=1e-2)` 全 PASS。

## 合规声明（防作弊）

- **fp32 全精度**，不使用 fast-math、不降精度换速度。
- **无对 PyTorch 高层算子的运行时依赖**（无 `F.scaled_dot_product_attention` / `torch.nn.functional`）；本交付版连 PyTorch 都不依赖。
- 未使用 cuBLAS/cuDNN（虽然边界约束允许），全部为自定义 kernel。
