import torch

from cases.rmsnorm import config


def reference_forward(inputs, params):
    x = inputs["x"]
    gamma = inputs["gamma"]
    eps = float(params["eps"])

    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    y = (x / rms) * gamma
    return y


def make_inputs(seed, dtype, device, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(seed)

    x = torch.randn(
        config.M,
        config.N,
        dtype=dtype,
        device=device,
        generator=g,
    )
    gamma = torch.randn(
        config.N,
        dtype=dtype,
        device=device,
        generator=g,
    )

    if requires_grad:
        x.requires_grad_(True)
        gamma.requires_grad_(True)

    return {
        "x": x,
        "gamma": gamma,
    }
