"""Autograd wrapper for the batched Thomas CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "tridiag_ext",
        ["tridiag.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _TridiagFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lower, diag, upper, rhs):
        lower = lower.contiguous()
        diag = diag.contiguous()
        upper = upper.contiguous()
        rhs = rhs.contiguous()
        x, factors = _extension().tridiag_forward(lower, diag, upper, rhs)
        ctx.save_for_backward(upper, x, factors)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        upper, x, factors = ctx.saved_tensors
        return tuple(
            _extension().tridiag_backward(
                grad_output.contiguous(), upper, x, factors
            )
        )


def candidate(inputs, params):
    return _TridiagFunction.apply(
        inputs["lower"], inputs["diag"], inputs["upper"], inputs["rhs"]
    )


def forward_only(inputs, params):
    return _extension().tridiag_forward(
        inputs["lower"].contiguous(),
        inputs["diag"].contiguous(),
        inputs["upper"].contiguous(),
        inputs["rhs"].contiguous(),
    )[0]
