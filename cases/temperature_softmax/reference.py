"""Vectorized PyTorch reference for temperature-scaled softmax."""

import torch

from cases.temperature_softmax import config


def reference_forward(inputs, params):
    scores = inputs["scores"]
    temperature = params["temperature"]

    scaled = scores / temperature
    shifted = scaled - scaled.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(shifted)
    return exp_scores / exp_scores.sum(dim=-1, keepdim=True)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    scores = torch.randn(
        config.B,
        config.D,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if requires_grad:
        scores.requires_grad_(True)
    return {"scores": scores}
