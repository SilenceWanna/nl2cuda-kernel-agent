# 算法结构描述：Softmax 交叉熵损失（Softmax Cross-Entropy Loss）

> 这是**测试输入**：喂给待测 agent 的自然语言描述。agent 需据此自己写出 PyTorch 参考
> 实现（reference.py / config.py / __init__.py）与 CUDA 前反向 kernel（kernels/ + op.py），
> 不得直接落回 F.cross_entropy / F.log_softmax 等高层算子。

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
  - `dlogits[b,c] = (softmax(logits)[b,c] - onehot(labels[b])[c]) / B`

## Shape / dtype 约定

- B = 8192（样本数），C = 1024（类别数）
- logits: float32；labels: int64
- 输出 loss 为 float32 标量

## 提示（供实现参考，非约束）

- 前向数值稳定：logsumexp 要先减去每行最大值再 exp（避免溢出）。
- 反向异常简洁：`dlogits = (softmax - onehot) / B`，无需保存中间 softmax 也可（反向重算或前向缓存）。
- 这是本 skill 的**通用性压测点**：输出是**标量**（非矩阵/同形张量），输入含**整型 labels**（不求梯度）——
  与 RBF（双输入矩阵输出）、LayerNorm（同形输出）都不同。framework 的 Case 协议应能覆盖
  （grad_inputs 只含 "logits"；labels 放 inputs 但不在 grad_inputs；upstream 梯度对标量是 scalar）。
