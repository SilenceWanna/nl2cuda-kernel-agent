"""Autograd wrapper for the custom CUDA GroupNorm kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "groupnorm_ext",
        ["groupnorm.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _GroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, groups, eps):
        X = X.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        output, mean, rstd = _extension().groupnorm_forward(
            X, gamma, beta, int(groups), float(eps)
        )
        ctx.save_for_backward(X, gamma, mean, rstd)
        ctx.groups = int(groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, gamma, mean, rstd = ctx.saved_tensors
        grad_X, grad_gamma, grad_beta = _extension().groupnorm_backward(
            grad_output.contiguous(), X, gamma, mean, rstd, ctx.groups
        )
        return grad_X, grad_gamma, grad_beta, None, None


def candidate(inputs, params):
    return _GroupNormFunction.apply(
        inputs["X"],
        inputs["gamma"],
        inputs["beta"],
        int(params["groups"]),
        float(params["eps"]),
    )


def forward_only(inputs, params):
    output, _, _ = _extension().groupnorm_forward(
        inputs["X"].contiguous(),
        inputs["gamma"].contiguous(),
        inputs["beta"].contiguous(),
        int(params["groups"]),
        float(params["eps"]),
    )
    return output
