"""LayerNorm 的 PyTorch 参考实现（正确性金标准，dict 接口）。

结构：Y = (X - mean)/sqrt(var+eps) * gamma + beta，在最后一维 D 上归一化。

关键（防作弊）：**不落回 F.layer_norm / torch.nn.LayerNorm 等高层算子**，
用基础规约 + 广播 + 逐元素表达；反向由 autograd 提供。
"""

import torch

from cases.layernorm import config


def reference_forward(inputs, params):
    """inputs={"X":[B,D], "gamma":[D], "beta":[D]}, params={"eps":float} -> Y:[B,D]。"""
    X, gamma, beta = inputs["X"], inputs["gamma"], inputs["beta"]
    eps = params["eps"]
    mean = X.mean(dim=-1, keepdim=True)
    var = ((X - mean) ** 2).mean(dim=-1, keepdim=True)   # unbiased=False
    xhat = (X - mean) / torch.sqrt(var + eps)
    return xhat * gamma + beta


def make_inputs(seed, dtype, device, requires_grad=False):
    """按种子生成命名输入 {"X","gamma","beta"}。gamma/beta 是待求梯度的参数张量。"""
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(config.B, config.D, dtype=dtype, device=device, generator=g)
    gamma = torch.randn(config.D, dtype=dtype, device=device, generator=g)
    beta = torch.randn(config.D, dtype=dtype, device=device, generator=g)
    if requires_grad:
        X.requires_grad_(True)
        gamma.requires_grad_(True)
        beta.requires_grad_(True)
    return {"X": X, "gamma": gamma, "beta": beta}
