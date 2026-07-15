"""Autograd wrapper for the custom RoPE CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "rope_ext",
        ["rope.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, base):
        ctx.base = float(base)
        return _extension().rope_forward(X, ctx.base)

    @staticmethod
    def backward(ctx, grad_output):
        grad_X = _extension().rope_backward(grad_output.contiguous(), ctx.base)
        return grad_X, None


def candidate(inputs, params):
    return _RoPEFunction.apply(inputs["X"], params["base"])


def forward_only(inputs, params):
    return _extension().rope_forward(inputs["X"], float(params["base"]))
