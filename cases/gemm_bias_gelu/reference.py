import torch

from cases.gemm_bias_gelu import config


_GELU_C = 0.7978845608028654
_GELU_A = 0.044715


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(_GELU_C * (x + _GELU_A * x * x * x)))


def reference_forward(inputs, params):
    x = inputs["X"]
    w = inputs["W"]
    b = inputs["b"]
    z = x.matmul(w) + b
    return _gelu_tanh(z)


def make_inputs(seed, dtype, device, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(seed)

    x = torch.randn(
        config.M,
        config.K,
        dtype=dtype,
        device=device,
        generator=g,
    )
    w = torch.randn(
        config.K,
        config.N,
        dtype=dtype,
        device=device,
        generator=g,
    )
    b = torch.randn(
        config.N,
        dtype=dtype,
        device=device,
        generator=g,
    )

    if requires_grad:
        x.requires_grad_(True)
        w.requires_grad_(True)
        b.requires_grad_(True)

    return {
        "X": x,
        "W": w,
        "b": b,
    }
