"""Vectorized PyTorch gold standard for Welford LayerNorm.

The CUDA implementation computes the same population statistics with online
Welford updates. The reference deliberately uses equivalent whole-tensor
reductions so torch.compile sees an O(B*N*D), loop-free graph.
"""

import torch

from cases.welford import config


def reference_forward(inputs, params):
    X = inputs["X"]
    gamma = inputs["gamma"]
    beta = inputs["beta"]

    mean = X.mean(dim=-1, keepdim=True)
    centered = X - mean
    var = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered / torch.sqrt(var + params["eps"])
    return normalized * gamma + beta


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(
        config.B,
        config.N,
        config.D,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    gamma = torch.randn(
        config.D, dtype=dtype, device=device, generator=generator
    )
    beta = torch.randn(
        config.D, dtype=dtype, device=device, generator=generator
    )

    if requires_grad:
        X.requires_grad_(True)
        gamma.requires_grad_(True)
        beta.requires_grad_(True)

    return {"X": X, "gamma": gamma, "beta": beta}
