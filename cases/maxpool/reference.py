"""Vectorized PyTorch reference for 2x2 stride-2 max pooling."""

import torch

from cases.maxpool import config


def reference_forward(inputs, params):
    X = inputs["X"]
    N, C, H, W = X.shape
    windows = X.reshape(N, C, H // 2, 2, W // 2, 2)
    return windows.max(dim=5).values.max(dim=3).values


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
    if requires_grad:
        X.requires_grad_(True)
    return {"X": X}
