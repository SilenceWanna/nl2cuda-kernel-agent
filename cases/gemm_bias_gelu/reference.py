import math

import torch

from cases.gemm_bias_gelu import config


_GELU_COEFF = math.sqrt(2.0 / math.pi)


def reference_forward(inputs, params):
    x = inputs["X"]
    w = inputs["W"]
    b = inputs["b"]

    z = x @ w + b[None, :]
    cubic = z * z * z
    return 0.5 * z * (1.0 + torch.tanh(_GELU_COEFF * (z + 0.044715 * cubic)))


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        config.M,
        config.K,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    w = torch.randn(
        config.K,
        config.N,
        dtype=dtype,
        device=device,
        generator=generator,
    ) / math.sqrt(config.K)
    b = torch.randn(
        config.N,
        dtype=dtype,
        device=device,
        generator=generator,
    ) * 0.1

    if requires_grad:
        x.requires_grad_(True)
        w.requires_grad_(True)
        b.requires_grad_(True)
    return {"X": x, "W": w, "b": b}

