# GEMM + bias + GELU

给定 fp32 输入 `X[M,K]`、权重 `W[K,N]` 和偏置 `b[N]`，计算

\[
Z = XW + b, \qquad
Y = \operatorname{GELU}(Z),
\]

其中 GELU 使用 tanh 近似：

\[
\operatorname{GELU}(z)
= \frac{1}{2}z\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}
\left(z+0.044715z^3\right)\right)\right).
\]

输出 `Y[M,N]`，并对 `X`、`W`、`b` 求梯度。默认规模为
`M=8192, K=256, N=1024`；可分别通过 `GBG_M`、`GBG_K`、`GBG_N`
环境变量覆盖。

