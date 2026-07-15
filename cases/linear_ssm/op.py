"""Autograd wrapper for the custom linear SSM CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "linear_ssm_ext",
        ["linear_ssm.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _LinearSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b_coef):
        ctx.a = float(a)
        ctx.b_coef = float(b_coef)
        return _extension().linear_ssm_forward(x.contiguous(), ctx.a, ctx.b_coef)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = _extension().linear_ssm_backward(
            grad_output.contiguous(), ctx.a, ctx.b_coef
        )
        return grad_x, None, None


def candidate(inputs, params):
    return _LinearSSMFunction.apply(inputs["X"], params["a"], params["b_coef"])


def forward_only(inputs, params):
    return _extension().linear_ssm_forward(
        inputs["X"].contiguous(), float(params["a"]), float(params["b_coef"])
    )
