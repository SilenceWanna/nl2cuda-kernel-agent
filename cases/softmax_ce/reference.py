"""Softmax 交叉熵损失的 PyTorch 参考实现（正确性金标准，dict 接口）。

结构：logits:[B,C], labels:[B](int64) → 标量 loss = mean_b(-logp[b, labels[b]])
      logp = logits - logsumexp(logits, dim=-1)

关键（防作弊）：**不落回 F.cross_entropy / F.log_softmax / F.nll_loss 等高层算子**，
用基础算子（logsumexp 规约 + gather 索引 + mean）表达；反向由 autograd 提供。
"""

import torch

from cases.softmax_ce import config


def reference_forward(inputs, params):
    """inputs={"logits":[B,C], "labels":[B] int64} -> 标量 loss。"""
    logits, labels = inputs["logits"], inputs["labels"]
    # log-softmax 手写：logits - logsumexp（数值稳定由 logsumexp 内部处理）
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)      # [B,C]
    nll = -logp.gather(1, labels.unsqueeze(1)).squeeze(1)              # [B]
    return nll.mean()                                                  # 标量


def make_inputs(seed, dtype, device, requires_grad=False):
    """命名输入 {"logits","labels"}。labels 恒 int64、不求梯度（不受 dtype 影响）。"""
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(config.B, config.C, dtype=dtype, device=device, generator=g)
    labels = torch.randint(0, config.C, (config.B,), device=device, generator=g)
    if requires_grad:
        logits.requires_grad_(True)   # 只有 logits 求梯度；labels 是整型索引
    return {"logits": logits, "labels": labels}
