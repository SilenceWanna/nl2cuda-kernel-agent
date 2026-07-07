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

## 备注（供 kernel 实现参考，非约束）

- 前向朴素参考用广播形式 `(X[:,None,:] - Y[None,:,:])^2.sum(-1)`，会物化 [N,M,D] 大中间张量——
  手写融合 kernel 可避免物化，这是内存带宽优势来源。
- 反向数学：令 `S[i,j] = -gamma * G[i,j] * K[i,j]`（G 为上游梯度），则
  `dX[i] = sum_j S[i,j] * 2*(x_i - y_j)`，`dY[j] = sum_i S[i,j] * 2*(y_j - x_i)`。
