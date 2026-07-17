"""Vectorized PyTorch reference for scatter-add."""

import torch

from cases.scatter_add import config


def reference_forward(inputs, params):
    X = inputs["X"]
    idx = inputs["idx"]
    output = torch.zeros(
        (params["S"], X.size(1)), dtype=X.dtype, device=X.device
    )
    return output.index_add_(0, idx, X)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(
        config.N,
        config.D,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    idx = torch.randint(
        0,
        config.S,
        (config.N,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    if requires_grad:
        X.requires_grad_(True)
    return {"X": X, "idx": idx}
