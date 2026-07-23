"""Vectorized PyTorch reference for tanh-approximate GeGLU."""

import math

import torch

from cases.geglu import config


_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_CUBIC_COEFF = 0.044715


def reference_forward(inputs, params):
    """X[B, T, 2H] -> Y[B, T, H] using the GPT-style tanh GELU."""
    x = inputs["X"]
    h = x.shape[-1] // 2
    value = x[..., :h]
    gate = x[..., h:]
    cubic = gate * gate * gate
    u = _SQRT_2_OVER_PI * (gate + _CUBIC_COEFF * cubic)
    gelu_gate = 0.5 * gate * (1.0 + torch.tanh(u))
    return value * gelu_gate


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        config.B,
        config.T,
        2 * config.H,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if requires_grad:
        x.requires_grad_(True)
    return {"X": x}
