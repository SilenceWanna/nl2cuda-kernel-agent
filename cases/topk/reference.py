"""Vectorized PyTorch reference for row-wise Top-K."""

import torch

from cases.topk import config


def reference_forward(inputs, params):
    return torch.topk(
        inputs["X"],
        k=params["k"],
        dim=1,
        largest=True,
        sorted=True,
    ).values


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(
        config.N,
        config.D,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if requires_grad:
        X.requires_grad_(True)
    return {"X": X}
