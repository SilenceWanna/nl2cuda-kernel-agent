"""PyTorch reference for a linear SSM over the time dimension."""

import torch

from cases.linear_ssm import config


def reference_forward(inputs, params):
    x = inputs["X"]
    a = float(params["a"])
    b_coef = float(params["b_coef"])
    time = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
    powers = torch.pow(torch.tensor(a, dtype=x.dtype, device=x.device), time).view(1, -1, 1)
    inv_powers = torch.pow(torch.tensor(1.0 / a, dtype=x.dtype, device=x.device), time).view(1, -1, 1)
    return b_coef * powers * torch.cumsum(x * inv_powers, dim=1)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        config.B,
        config.T,
        config.C,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if requires_grad:
        x.requires_grad_(True)
    return {"X": x}
