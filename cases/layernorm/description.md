# 算法结构描述：LayerNorm（层归一化）

## 自然语言描述

对输入的每一行（最后一维）做层归一化：先减去该行均值、除以该行标准差（含 eps 稳定），
再用逐通道的可学习参数 gamma 缩放、beta 平移。这是 Transformer 等网络的标准组件。

## 数学定义

- 输入：X 形状 [B, D]（在最后一维 D 上归一化）
- 每行：`mean = (1/D) Σ_d X[b,d]`，`var = (1/D) Σ_d (X[b,d]-mean)^2`
- 归一化：`xhat[b,d] = (X[b,d] - mean_b) / sqrt(var_b + eps)`
- 输出：`Y[b,d] = xhat[b,d] * gamma[d] + beta[d]`，形状 [B, D]
- 反向：对 X、gamma、beta 均求梯度（dX、dgamma、dbeta）

## Shape / dtype 约定

- B = 4096（行数），D = 1024（归一化维度）
- dtype = float32
- eps = 1e-5
- gamma、beta 形状均为 [D]
