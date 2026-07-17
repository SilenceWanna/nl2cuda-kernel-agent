"""Autograd wrapper for the 2x2 stride-2 max-pooling CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "maxpool_ext",
        ["maxpool.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _MaxPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        Y = _extension().maxpool_forward(X)
        ctx.save_for_backward(X)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        (X,) = ctx.saved_tensors
        return _extension().maxpool_backward(X, grad_Y.contiguous())


def candidate(inputs, params):
    return _MaxPoolFunction.apply(inputs["X"])


def forward_only(inputs, params):
    return _extension().maxpool_forward(inputs["X"])
