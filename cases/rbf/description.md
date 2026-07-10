# 算法结构描述：RBF 高斯核矩阵（成对距离）

## 自然语言描述

给定两组向量 X 和 Y，计算它们之间的 RBF（径向基函数 / 高斯）核矩阵。
对 X 中每个向量 x_i 和 Y 中每个向量 y_j，先算它们的平方欧氏距离，
再取高斯核 `exp(-gamma * 距离)`。输出是一个 N×M 的核矩阵 K。

## 数学定义

- 输入：X 形状 [N, D]，Y 形状 [M, D]
- 距离：`dist[i,j] = sum_d (X[i,d] - Y[j,d])^2`
- 输出：`K[i,j] = exp(-gamma * dist[i,j])`，形状 [N, M]
- 反向：对 X 和 Y 均求梯度（dX、dY）

## Shape / dtype 约定

- N = M = 2048（默认；env `RBF_SIZE=4096` 可切大形状），D = 64
- dtype = float32
- gamma = 1/64
