import torch

from cases.conv1d import config


def _shift_right(x, amount):
    zeros = torch.zeros_like(x[:, :, :amount])
    return torch.cat((zeros, x[:, :, :-amount]), dim=2)


def reference_forward(inputs, params):
    x = inputs["X"]
    w = inputs["W"]
    return (
        x * w[:, 0].view(1, -1, 1)
        + _shift_right(x, 1) * w[:, 1].view(1, -1, 1)
        + _shift_right(x, 2) * w[:, 2].view(1, -1, 1)
        + _shift_right(x, 3) * w[:, 3].view(1, -1, 1)
    )


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        config.B,
        config.C,
        config.T,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    w = torch.randn(
        config.C,
        config.K,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if requires_grad:
        x.requires_grad_(True)
        w.requires_grad_(True)
    return {"X": x, "W": w}
