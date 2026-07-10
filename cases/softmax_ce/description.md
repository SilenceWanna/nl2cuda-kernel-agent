# 算法结构描述：Softmax 交叉熵损失（Softmax Cross-Entropy Loss）

## 自然语言描述

给定一批样本的分类打分（logits）和每个样本的正确类别标签，计算 softmax 交叉熵损失。
对每个样本，先对它的 C 个类别打分做 softmax 得到概率分布，再取正确类别概率的负对数；
对整批样本的损失取平均，得到一个标量 loss。这是分类任务最常用的损失函数。

## 数学定义

- 输入：`logits` 形状 `[B, C]`（float32，B 个样本、C 个类别）；`labels` 形状 `[B]`（int64，取值 0..C-1）
- 每样本：`logp[b,c] = logits[b,c] - logsumexp_c(logits[b,:])`
- 每样本损失：`loss_b = -logp[b, labels[b]]`
- 输出：标量 `loss = mean_b(loss_b)`（对 batch 取平均）
- 反向：对 `logits` 求梯度 `dlogits`（`labels` 为整型索引，不求梯度）

## Shape / dtype 约定

- B = 8192（样本数），C = 1024（类别数）
- logits: float32；labels: int64
- 输出 loss 为 float32 标量
