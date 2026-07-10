import torch

from cases.layernorm import config


def reference_forward(inputs, params):
    x = inputs["X"]
    gamma = inputs["gamma"]
    beta = inputs["beta"]
    eps = params.get("eps", config.EPS)

    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    xhat = centered * torch.rsqrt(var + eps)
    return xhat * gamma + beta


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
        config.B,
        config.D,
        dtype=dtype,
        device=device,
        generator=g,
    )

    gamma = 1.0 + 0.02 * torch.randn(
        config.D,
        dtype=dtype,
        device=device,
        generator=g,
    )

    beta = 0.02 * torch.randn(
        config.D,
        dtype=dtype,
        device=device,
        generator=g,
    )

    if requires_grad:
        x.requires_grad_(True)
        gamma.requires_grad_(True)
        beta.requires_grad_(True)

    return {
        "X": x,
        "gamma": gamma,
        "beta": beta,
    }
