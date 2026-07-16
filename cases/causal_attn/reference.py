"""PyTorch reference for single-head causal self-attention."""

import math

import torch

from cases.causal_attn import config


def reference_forward(inputs, params):
    Q, K, V = inputs["Q"], inputs["K"], inputs["V"]
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(d))
    mask = torch.ones((scores.shape[-2], scores.shape[-1]), device=scores.device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, V)


def make_inputs(seed, dtype, device, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(config.B, config.T, config.D, dtype=dtype, device=device, generator=g)
    K = torch.randn(config.B, config.T, config.D, dtype=dtype, device=device, generator=g)
    V = torch.randn(config.B, config.T, config.D, dtype=dtype, device=device, generator=g)
    if requires_grad:
        Q.requires_grad_(True)
        K.requires_grad_(True)
        V.requires_grad_(True)
    return {"Q": Q, "K": K, "V": V}
