"""Autograd wrapper for depthwise causal Conv1d CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "conv1d_ext",
        ["conv1d.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _Conv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        y = _extension().conv1d_forward(x, w)
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x, grad_w = _extension().conv1d_backward(grad_output.contiguous(), x, w)
        return grad_x, grad_w


def candidate(inputs, params):
    return _Conv1dFunction.apply(inputs["X"], inputs["W"])


def forward_only(inputs, params):
    return _extension().conv1d_forward(inputs["X"], inputs["W"])
