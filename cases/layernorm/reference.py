"""PyTorch reference implementation for the LayerNorm case."""

import torch

from cases.layernorm import config


def reference_forward(inputs, params):
    """LayerNorm forward written with basic tensor ops.

    Inputs:
      X:     [B, D] fp32
      gamma: [D] fp32
      beta:  [D] fp32
    Output:
      Y:     [B, D] fp32
    """
    x = inputs["X"]
    gamma = inputs["gamma"]
    beta = inputs["beta"]
    eps = float(params.get("eps", config.EPS))

    mean = x.mean(dim=1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    xhat = centered * inv_std
    return xhat * gamma.unsqueeze(0) + beta.unsqueeze(0)


def make_inputs(seed, dtype, device, requires_grad=False):
    """Create deterministic LayerNorm inputs.

    Only tensors listed in CASE.grad_inputs get requires_grad=True.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(config.B, config.D, dtype=dtype, device=device, generator=g)
    gamma = torch.randn(config.D, dtype=dtype, device=device, generator=g)
    beta = torch.randn(config.D, dtype=dtype, device=device, generator=g)

    if requires_grad:
        x.requires_grad_(True)
        gamma.requires_grad_(True)
        beta.requires_grad_(True)

    return {"X": x, "gamma": gamma, "beta": beta}
