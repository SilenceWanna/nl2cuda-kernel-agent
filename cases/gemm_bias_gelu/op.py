import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "gemm_bias_gelu_ext",
        ["gemm_bias_gelu.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _GemmBiasGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        x = x.contiguous()
        w = w.contiguous()
        b = b.contiguous()
        out, gemm_output = _extension().gemm_bias_gelu_forward(x, w, b)
        ctx.save_for_backward(x, w, b, gemm_output)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w, b, gemm_output = ctx.saved_tensors
        grad_x, grad_w, grad_b = _extension().gemm_bias_gelu_backward(
            grad_out.contiguous(), x, w, b, gemm_output
        )
        return grad_x, grad_w, grad_b


def candidate(inputs, params):
    return _GemmBiasGeluFunction.apply(inputs["X"], inputs["W"], inputs["b"])


def forward_only(inputs, params):
    out, _ = _extension().gemm_bias_gelu_forward(
        inputs["X"].contiguous(),
        inputs["W"].contiguous(),
        inputs["b"].contiguous(),
    )
    return out
