import torch

from cases.layernorm import config


def reference_forward(inputs, params):
    x = inputs["X"]
    gamma = inputs["gamma"]
    beta = inputs["beta"]
    eps = float(params["eps"])

    mean = x.mean(dim=1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=1, keepdim=True)
    xhat = centered * torch.rsqrt(var + eps)
    return xhat * gamma[None, :] + beta[None, :]


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        config.B,
        config.D,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    gamma = torch.randn(config.D, dtype=dtype, device=device, generator=generator)
    beta = torch.randn(config.D, dtype=dtype, device=device, generator=generator)

    if requires_grad:
        x.requires_grad_(True)
        gamma.requires_grad_(True)
        beta.requires_grad_(True)

    return {"X": x, "gamma": gamma, "beta": beta}
