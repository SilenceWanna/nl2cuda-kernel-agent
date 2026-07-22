"""Vectorized PyTorch reference for NCHW GroupNorm."""

import torch

from cases.groupnorm import config


def reference_forward(inputs, params):
    """Normalize each sample/group over its channels and spatial positions."""
    X = inputs["X"]
    gamma = inputs["gamma"]
    beta = inputs["beta"]
    groups = params["groups"]
    eps = params["eps"]

    n, c, h, w = X.shape
    grouped = X.reshape(n, groups, c // groups, h, w)
    mean = grouped.mean(dim=(2, 3, 4), keepdim=True)
    centered = grouped - mean
    var = (centered * centered).mean(dim=(2, 3, 4), keepdim=True)
    xhat = (centered * torch.rsqrt(var + eps)).reshape_as(X)
    return xhat * gamma.reshape(1, c, 1, 1) + beta.reshape(1, c, 1, 1)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(
        config.N,
        config.C,
        config.H,
        config.W,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    gamma = torch.randn(config.C, dtype=dtype, device=device, generator=generator)
    beta = torch.randn(config.C, dtype=dtype, device=device, generator=generator)
    if requires_grad:
        X.requires_grad_(True)
        gamma.requires_grad_(True)
        beta.requires_grad_(True)
    return {"X": X, "gamma": gamma, "beta": beta}
