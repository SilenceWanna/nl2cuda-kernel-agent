"""Autograd wrapper for the bilinear grid-sample CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel("gridsample_ext", ["gridsample.cu"],
                       base_dir=_KERNEL_DIR, verbose=False)


class _GridSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, grid):
        Y = _extension().gridsample_forward(X, grid)
        ctx.save_for_backward(grid)
        ctx.input_h = X.size(2)
        ctx.input_w = X.size(3)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        (grid,) = ctx.saved_tensors
        grad_X = _extension().gridsample_backward(
            grid, grad_Y.contiguous(), ctx.input_h, ctx.input_w)
        return grad_X, None


def candidate(inputs, params):
    return grid_sample(inputs["X"], inputs["grid"])


def grid_sample(X, grid):
    """Sample ``X[N,C,H,W]`` at normalized ``grid[N,OH,OW,2]`` coordinates."""
    return _GridSampleFunction.apply(X, grid)


def forward_only(inputs, params):
    return _extension().gridsample_forward(inputs["X"], inputs["grid"])
