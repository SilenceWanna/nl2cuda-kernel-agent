import torch

from cases.rmsnorm import config


def reference_forward(inputs, params):
    x = inputs["X"]
    gamma = inputs["gamma"]
    eps = float(params.get("eps", config.EPS))

    mean_square = (x * x).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    return x * inv_rms * gamma


def make_inputs(seed, dtype, device, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(seed)

    x = torch.randn(
        config.B,
        config.D,
        dtype=dtype,
        device=device,
        generator=g,
    )
    gamma = torch.randn(
        config.D,
        dtype=dtype,
        device=device,
        generator=g,
    )

    if requires_grad:
        x.requires_grad_(True)
        gamma.requires_grad_(True)

    return {
        "X": x,
        "gamma": gamma,
    }
