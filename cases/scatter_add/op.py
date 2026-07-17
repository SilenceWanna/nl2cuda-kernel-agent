"""Autograd wrapper for the scatter-add CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "scatter_add_ext",
        ["scatter_add.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _ScatterAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, idx, segments):
        idx = idx.contiguous()
        ctx.save_for_backward(idx)
        return _extension().scatter_add_forward(X.contiguous(), idx, int(segments))

    @staticmethod
    def backward(ctx, grad_output):
        (idx,) = ctx.saved_tensors
        grad_X = _extension().scatter_add_backward(grad_output.contiguous(), idx)
        return grad_X, None, None


def candidate(inputs, params):
    return _ScatterAddFunction.apply(inputs["X"], inputs["idx"], params["S"])


def forward_only(inputs, params):
    return _extension().scatter_add_forward(
        inputs["X"].contiguous(), inputs["idx"].contiguous(), int(params["S"])
    )
