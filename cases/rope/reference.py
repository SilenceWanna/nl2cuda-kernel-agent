"""Vectorized PyTorch reference for rotary position embedding."""

import math

import torch

from cases.rope import config


def reference_forward(inputs, params):
    X = inputs["X"]
    base = params["base"]
    _, S, _, D = X.shape

    pair_index = torch.arange(D // 2, dtype=X.dtype, device=X.device)
    inv_freq = torch.exp(pair_index * (-2.0 * math.log(base) / D))
    position = torch.arange(S, dtype=X.dtype, device=X.device)
    theta = position[:, None] * inv_freq[None, :]
    cos_theta = torch.cos(theta).reshape(1, S, 1, D // 2)
    sin_theta = torch.sin(theta).reshape(1, S, 1, D // 2)

    pairs = X.reshape(*X.shape[:-1], D // 2, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    rotated = torch.stack(
        (even * cos_theta - odd * sin_theta,
         even * sin_theta + odd * cos_theta),
        dim=-1,
    )
    return rotated.reshape_as(X)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(
        config.B,
        config.S,
        config.H,
        config.D,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if requires_grad:
        X.requires_grad_(True)
    return {"X": X}
