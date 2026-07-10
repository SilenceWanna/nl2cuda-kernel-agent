import torch

from cases.rbf import config


def reference_forward(inputs, params):
    x = inputs["X"]
    y = inputs["Y"]
    gamma = float(params["gamma"])
    diff = x[:, None, :] - y[None, :, :]
    dist = (diff * diff).sum(dim=2)
    return torch.exp(-gamma * dist)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(config.N, config.D, dtype=dtype, device=device, generator=generator)
    y = torch.randn(config.M, config.D, dtype=dtype, device=device, generator=generator)
    if requires_grad:
        x.requires_grad_(True)
        y.requires_grad_(True)
    return {"X": x, "Y": y}
