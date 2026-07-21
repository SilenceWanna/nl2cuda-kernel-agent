"""l2norm_scale 的 PyTorch 参考实现（正确性金标准，dict 接口）。

L2 归一化（每行除以自身 L2 模长）+ 每维度可学习权重缩放。
用基础规约+广播表达，不落回 F.normalize 等高层算子；autograd 提供反向。
"""

import torch

from cases.l2norm_scale import config


def reference_forward(inputs, params):
    x = inputs["X"]              # [N, D]
    g = inputs["g"]              # [D]
    eps = params["eps"]
    norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)  # [N,1] 每行 L2 模长
    return (x / norm) * g        # 归一化后逐维缩放（g 广播）


def make_inputs(seed, dtype, device, requires_grad=False):
    g_ = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(config.N, config.D, dtype=dtype, device=device, generator=g_)
    g = torch.randn(config.D, dtype=dtype, device=device, generator=g_)
    if requires_grad:
        x.requires_grad_(True)
        g.requires_grad_(True)
    return {"X": x, "g": g}
