# RMSNorm

实现 RMSNorm 前向与反向 CUDA kernel。

## 输入

- `X`: shape `[4096, 1024]`, dtype `float32`
- `gamma`: shape `[1024]`, dtype `float32`
- `eps`: `1e-5`

## 前向

对每一行 `b`：

```text
rms[b] = sqrt(mean_j(X[b, j]^2) + eps)
Y[b, j] = X[b, j] / rms[b] * gamma[j]
```

等价地：

```text
inv_rms[b] = rsqrt(mean_j(X[b, j]^2) + eps)
Y[b, j] = X[b, j] * inv_rms[b] * gamma[j]
```

## 反向

对 `X` 和 `gamma` 求梯度。

给定上游梯度 `dY`：

```text
dgamma[j] = sum_b dY[b, j] * X[b, j] * inv_rms[b]

dot[b] = sum_j dY[b, j] * gamma[j] * X[b, j]

dX[b, i] = dY[b, i] * gamma[i] * inv_rms[b]
           - X[b, i] * inv_rms[b]^3 / D * dot[b]
```

其中 `D = 1024`。
