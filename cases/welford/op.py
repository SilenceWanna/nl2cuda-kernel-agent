"""Autograd wrapper for the custom CUDA Welford LayerNorm kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "welford_ext",
        ["welford.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _WelfordLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, eps):
        X = X.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        output, mean, rstd = _extension().welford_forward(
            X, gamma, beta, float(eps)
        )
        ctx.save_for_backward(X, gamma, mean, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, gamma, mean, rstd = ctx.saved_tensors
        grad_X, grad_gamma, grad_beta = _extension().welford_backward(
            grad_output.contiguous(), X, gamma, mean, rstd
        )
        return grad_X, grad_gamma, grad_beta, None


def candidate(inputs, params):
    return _WelfordLayerNormFunction.apply(
        inputs["X"], inputs["gamma"], inputs["beta"], float(params["eps"])
    )


def forward_only(inputs, params):
    output, _, _ = _extension().welford_forward(
        inputs["X"].contiguous(),
        inputs["gamma"].contiguous(),
        inputs["beta"].contiguous(),
        float(params["eps"]),
    )
    return output
