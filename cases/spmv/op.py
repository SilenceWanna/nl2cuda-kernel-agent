"""Autograd wrapper for the CSR SpMM/SpMV CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "spmv_ext",
        ["spmv.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _CSRSpMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, row_ptr, col_idx, vals, X):
        row_ptr = row_ptr.contiguous()
        col_idx = col_idx.contiguous()
        vals = vals.contiguous()
        X = X.contiguous()
        ctx.save_for_backward(row_ptr, col_idx, vals, X)
        return _extension().spmv_forward(row_ptr, col_idx, vals, X)

    @staticmethod
    def backward(ctx, grad_output):
        row_ptr, col_idx, vals, X = ctx.saved_tensors
        grad_vals, grad_X = _extension().spmv_backward(
            row_ptr, col_idx, vals, X, grad_output.contiguous()
        )
        return None, None, grad_vals, grad_X


def candidate(inputs, params):
    del params
    return _CSRSpMMFunction.apply(
        inputs["row_ptr"], inputs["col_idx"], inputs["vals"], inputs["X"]
    )


def forward_only(inputs, params):
    del params
    return _extension().spmv_forward(
        inputs["row_ptr"].contiguous(),
        inputs["col_idx"].contiguous(),
        inputs["vals"].contiguous(),
        inputs["X"].contiguous(),
    )
