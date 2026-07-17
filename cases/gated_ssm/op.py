"""Autograd wrapper for the gated SSM CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "gated_ssm_ext",
        ["gated_ssm.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _GatedSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        x = x.contiguous()
        w = w.contiguous()
        b = b.contiguous()
        y = _extension().gated_ssm_forward(x, w, b)
        ctx.save_for_backward(x, w, b, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, w, b, y = ctx.saved_tensors
        grad_x, grad_w, grad_b = _extension().gated_ssm_backward(
            grad_output.contiguous(), x, w, b, y
        )
        return grad_x, grad_w, grad_b


def candidate(inputs, params):
    return _GatedSSMFunction.apply(inputs["X"], inputs["w"], inputs["b"])


def forward_only(inputs, params):
    return _extension().gated_ssm_forward(
        inputs["X"].contiguous(),
        inputs["w"].contiguous(),
        inputs["b"].contiguous(),
    )
