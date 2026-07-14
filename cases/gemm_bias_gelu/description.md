# GEMM + bias + GELU 融合

实现融合前向：

```
Z = X @ W + b
Y = gelu(Z)
```

其中：

- `X` 形状为 `[M, K]`，fp32
- `W` 形状为 `[K, N]`，fp32
- `b` 形状为 `[N]`，fp32
- 输出 `Y` 形状为 `[M, N]`，fp32

GELU 使用 tanh 近似形式：

```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

需要对 `X`、`W`、`b` 求梯度。
