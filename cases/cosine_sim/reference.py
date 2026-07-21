"""成对余弦相似度矩阵的 PyTorch 参考实现（金标准，dict 接口）。

S[i,j] = (A[i]·B[j]) / (|A[i]| · |B[j]|)，A[N,D] × B[M,D] → S[N,M]。
用广播式表达（归一化后逐元素乘再沿 D 规约），不落回 cdist/GEMM 分解——
torch.compile 对广播式生成通用 Triton（缺 GEMM 式 D 倍复用），可被手写 tiling 打败。
反向由 autograd 提供。
"""

import torch

from cases.cosine_sim import config


def reference_forward(inputs, params):
    A = inputs["A"]                                   # [N, D]
    B = inputs["B"]                                   # [M, D]
    eps = params["eps"]
    Ah = A / (A.norm(dim=-1, keepdim=True) + eps)     # [N, D] 单位化
    Bh = B / (B.norm(dim=-1, keepdim=True) + eps)     # [M, D]
    # 广播式：S[i,j] = sum_d Ah[i,d]*Bh[j,d]（不写成 Ah @ Bh.T 以免走 cuBLAS）
    return (Ah[:, None, :] * Bh[None, :, :]).sum(dim=-1)   # [N, M]


def make_inputs(seed, dtype, device, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(config.N, config.D, dtype=dtype, device=device, generator=g)
    B = torch.randn(config.M, config.D, dtype=dtype, device=device, generator=g)
    if requires_grad:
        A.requires_grad_(True)
        B.requires_grad_(True)
    return {"A": A, "B": B}
