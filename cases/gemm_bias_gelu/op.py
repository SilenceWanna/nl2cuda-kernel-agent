import os

import torch

from framework.loader import load_kernel


_MOD = None


def _load_module():
    global _MOD
    if _MOD is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        _MOD = load_kernel(
            "gemm_bias_gelu",
            sources=[
                os.path.join(base_dir, "kernels", "gemm_bias_gelu.cu"),
            ],
            base_dir=base_dir,
        )
    return _MOD


class _GemmBiasGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        mod = _load_module()

        x_c = x.contiguous()
        w_c = w.contiguous()
        b_c = b.contiguous()

        y, z = mod.gemm_bias_gelu_forward(x_c, w_c, b_c)
        ctx.save_for_backward(x_c, w_c, z)

        return y

    @staticmethod
    def backward(ctx, grad_y):
        mod = _load_module()

        x, w, z = ctx.saved_tensors
        grad_y_c = grad_y.contiguous()

        grad_x, grad_w, grad_b = mod.gemm_bias_gelu_backward(grad_y_c, x, w, z)

        return grad_x, grad_w, grad_b


def candidate(inputs, params):
    return _GemmBiasGeluFunction.apply(inputs["X"], inputs["W"], inputs["b"])


def forward_only(inputs, params):
    mod = _load_module()

    x_c = inputs["X"].contiguous()
    w_c = inputs["W"].contiguous()
    b_c = inputs["b"].contiguous()

    y, _ = mod.gemm_bias_gelu_forward(x_c, w_c, b_c)
    return y
