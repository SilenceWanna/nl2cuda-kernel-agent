"""Vectorized PyTorch reference for the input-dependent gated SSM."""

import torch

from cases.gated_ssm import config


def reference_forward(inputs, params):
    x = inputs["X"]
    w = inputs["w"]
    b = inputs["b"]
    logits = x * w.view(1, 1, -1) + b.view(1, 1, -1)

    # A cumprod/cumsum quotient underflows for natural gates over long sequences.
    # Pairwise log-prefix differences give the stable, vectorized O(T^2)
    # reference permitted for genuinely input-dependent recurrences.
    log_z = -torch.logaddexp(torch.zeros_like(logits), -logits)
    z = torch.exp(log_z)
    log_prefix = torch.cumsum(log_z, dim=1)
    log_transition = log_prefix[:, :, None, :] - log_prefix[:, None, :, :]

    time = torch.arange(x.shape[1], device=x.device)
    causal = time[:, None] >= time[None, :]
    masked_log_transition = torch.where(
        causal.view(1, x.shape[1], x.shape[1], 1),
        log_transition,
        torch.full((), -torch.inf, dtype=x.dtype, device=x.device),
    )
    transition = torch.exp(masked_log_transition)
    source = (1.0 - z) * x
    return (transition * source[:, None, :, :]).sum(dim=2)


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
    w = torch.randn(config.C, dtype=dtype, device=device, generator=generator)
    b = torch.randn(config.C, dtype=dtype, device=device, generator=generator)
    if requires_grad:
        x.requires_grad_(True)
        w.requires_grad_(True)
        b.requires_grad_(True)
    return {"X": x, "w": w, "b": b}
