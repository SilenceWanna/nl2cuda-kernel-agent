# 算法结构描述：Welford 单遍在线 LayerNorm

## 前向

输入 `X` 形状为 `[B,N,D]`，每个 `[D]` 行相互独立。沿最后一维用 Welford 在线更新：

- `count_n = n`
- `delta_n = x_n - mean_(n-1)`
- `mean_n = mean_(n-1) + delta_n / n`
- `delta2_n = x_n - mean_n`
- `M2_n = M2_(n-1) + delta_n * delta2_n`，`n = 1..D`

最终 `mean = mean_D`，`var = M2_D / D`，并计算：

`Y = gamma * (X - mean) / sqrt(var + eps) + beta`

CUDA 实现允许用 Welford 的标准 pairwise merge 公式并行合并各线程的在线状态，仍只对统计输入做一遍读取。

## Shape / dtype / 梯度

- 默认 `B=32, N=128, D=1024`，可分别由 `WELFORD_B`、`WELFORD_N`、`WELFORD_D` 覆盖
- `X: [B,N,D]`，`gamma: [D]`，`beta: [D]`
- `Y: [B,N,D]`
- `X`、`gamma`、`beta` 和所有中间统计均为 fp32，不使用 fast-math
- `eps = 1e-5`，方差采用总体方差 `M2_D / D`
- 对 `X`、`gamma`、`beta` 全部求梯度
