"""PyTorch correctness reference for an inclusive last-dimension scan."""

import torch

from cases.scan import config


def reference_forward(inputs, params):
    return torch.cumsum(inputs["X"], dim=-1)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        config.B,
        config.D,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if requires_grad:
        x.requires_grad_(True)
    return {"X": x}

