import torch

from cases.rbf import config


def reference_forward(inputs, params):
    x = inputs["X"]
    y = inputs["Y"]
    gamma = params.get("gamma", config.GAMMA)

    diff = x[:, None, :] - y[None, :, :]
    dist = (diff * diff).sum(dim=-1)
    return torch.exp(-gamma * dist)


def _resolve_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def make_inputs(seed, dtype, device, requires_grad=False):
    dtype = _resolve_dtype(dtype)
    g = torch.Generator(device=device).manual_seed(seed)

    x = torch.randn(
        config.N,
        config.D,
        dtype=dtype,
        device=device,
        generator=g,
    )

    y = torch.randn(
        config.M,
        config.D,
        dtype=dtype,
        device=device,
        generator=g,
    )

    if requires_grad:
        x.requires_grad_(True)
        y.requires_grad_(True)

    return {
        "X": x,
        "Y": y,
    }
