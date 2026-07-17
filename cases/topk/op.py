"""Autograd wrapper for the fixed-k row-wise Top-K CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "topk_ext",
        ["topk.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _TopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        values, indices = _extension().topk_forward(X)
        ctx.save_for_backward(indices)
        ctx.input_width = X.shape[1]
        return values

    @staticmethod
    def backward(ctx, grad_values):
        (indices,) = ctx.saved_tensors
        grad_X = _extension().topk_backward(
            grad_values.contiguous(), indices, ctx.input_width
        )
        return grad_X


def candidate(inputs, params):
    if params["k"] != 8:
        raise ValueError("the custom Top-K kernel requires k=8")
    return _TopKFunction.apply(inputs["X"])


def forward_only(inputs, params):
    if params["k"] != 8:
        raise ValueError("the custom Top-K kernel requires k=8")
    return _extension().topk_forward(inputs["X"])[0]
