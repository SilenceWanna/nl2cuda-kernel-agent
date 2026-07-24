"""Autograd wrapper for the blocked Cholesky CUDA implementation."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "cholesky_ext",
        ["cholesky.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _CholeskyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        factor = _extension().cholesky_forward(a.contiguous())
        ctx.save_for_backward(factor)
        return factor

    @staticmethod
    def backward(ctx, grad_output):
        (factor,) = ctx.saved_tensors
        grad_a = _extension().cholesky_backward(
            grad_output.contiguous(),
            factor,
        )
        return grad_a


def candidate(inputs, params):
    return _CholeskyFunction.apply(inputs["A"])


def forward_only(inputs, params):
    return _extension().cholesky_forward(inputs["A"].contiguous())
