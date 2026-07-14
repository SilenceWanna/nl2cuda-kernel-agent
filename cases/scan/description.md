# Inclusive Prefix-Sum

输入 `X[B, D]`，其中默认 `B = 4096`、`D = 4096`，dtype 为 `float32`。
沿最后一维计算包含式前缀和：

\[
y[b, i] = \sum_{j=0}^{i} x[b, j].
\]

输出 `Y` 的 shape 与 `X` 相同。仅对输入 `X` 求梯度；给定上游梯度
`dY`，反向为反向包含式前缀和：

\[
dX[b, j] = \sum_{i=j}^{D-1} dY[b, i].
\]

