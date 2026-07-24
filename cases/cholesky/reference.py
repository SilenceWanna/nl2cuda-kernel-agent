"""cuSOLVER-backed correctness and torch.compile reference."""

import torch

from cases.cholesky import config


def reference_forward(inputs, params):
    factor, _ = torch.linalg.cholesky_ex(
        inputs["A"],
        upper=False,
        check_errors=False,
    )
    return factor


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (config.N, config.N),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    a = (x @ x.transpose(0, 1)) / config.N
    a.diagonal().add_(1.0)
    a = a.detach()
    if requires_grad:
        a.requires_grad_(True)
    return {"A": a}
