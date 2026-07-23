"""Autograd wrapper for the tanh-GeGLU CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "geglu_ext",
        ["geglu.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _GeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        output = _extension().geglu_forward(x)
        # Backward intentionally recomputes tanh from X. This avoids an
        # output-sized auxiliary tensor in every training forward.
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = _extension().geglu_backward(x, grad_output.contiguous())
        return grad_x


def candidate(inputs, params):
    return _GeGLUFunction.apply(inputs["X"])


def forward_only(inputs, params):
    return _extension().geglu_forward(inputs["X"].contiguous())
