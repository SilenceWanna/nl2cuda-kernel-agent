"""PyTorch reference implementation for the RBF Gaussian kernel matrix case."""

import torch

from cases.rbf import config


def reference_forward(inputs, params):
    """Compute K[i, j] = exp(-gamma * sum_d (X[i, d] - Y[j, d])^2).

    This intentionally uses basic PyTorch tensor operations only. Autograd supplies
    the reference gradients for X and Y.
    """
    X = inputs["X"]
    Y = inputs["Y"]
    gamma = float(params["gamma"])

    diff = X[:, None, :] - Y[None, :, :]
    dist = (diff * diff).sum(dim=2)
    return torch.exp(-gamma * dist)


def make_inputs(seed, dtype, device, requires_grad=False):
    """Return named inputs for the RBF case.

    Only X and Y are differentiable inputs, matching CASE.grad_inputs.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(config.N, config.D, dtype=dtype, device=device, generator=g)
    Y = torch.randn(config.M, config.D, dtype=dtype, device=device, generator=g)
    if requires_grad:
        X.requires_grad_(True)
        Y.requires_grad_(True)
    return {"X": X, "Y": Y}
