# Gated SSM

输入 `X` 的形状为 `[B, T, C]`，门控参数 `w` 和 `b` 的形状均为 `[C]`，所有张量均为 fp32。
每个 batch、每个通道独立地沿时序执行输入依赖的变系数递推：

```text
z[b,t,c] = sigmoid(w[c] * X[b,t,c] + b[c])
h[b,t,c] = z[b,t,c] * h[b,t-1,c] + (1 - z[b,t,c]) * X[b,t,c]
```

初始状态 `h[b,-1,c] = 0`，输出 `Y[b,t,c] = h[b,t,c]`。对 `X`、`w`、`b` 求梯度。
