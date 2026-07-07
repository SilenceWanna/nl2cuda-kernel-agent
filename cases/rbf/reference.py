"""RBF 高斯核矩阵的 PyTorch 参考实现（正确性金标准，dict 接口）。

结构：X:[N,D], Y:[M,D] → K[i,j] = exp(-gamma * ||x_i - y_j||^2)

关键（赢面前提）：前向用**自然广播形式**，绝不写成 ||x||^2+||y||^2-2XY^T 的 GEMM 分解
（后者会让 torch.compile 走 cuBLAS，几乎无法被手写 kernel 打败）。反向由 autograd 提供。
"""

import torch

from cases.rbf import config


def reference_forward(inputs, params):
    """inputs={"X":[N,D], "Y":[M,D]}, params={"gamma":float} -> K:[N,M]（广播形式）。"""
    X, Y = inputs["X"], inputs["Y"]
    gamma = params["gamma"]
    diff = X.unsqueeze(1) - Y.unsqueeze(0)      # [N,M,D]，物化大中间量（手写 kernel 要避免）
    dist_sq = diff.pow(2).sum(dim=-1)           # [N,M]
    return torch.exp(-gamma * dist_sq)


def make_inputs(seed, dtype, device, requires_grad=False):
    """按种子生成命名输入 {"X","Y"}。"""
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(config.N, config.D, dtype=dtype, device=device, generator=g)
    Y = torch.randn(config.M, config.D, dtype=dtype, device=device, generator=g)
    if requires_grad:
        X.requires_grad_(True)
        Y.requires_grad_(True)
    return {"X": X, "Y": Y}
