import os

import torch

from framework.loader import load_kernel


_MOD = None


def _load_module():
    global _MOD
    if _MOD is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        _MOD = load_kernel(
            "rmsnorm",
            [
                "kernels/rmsnorm_forward.cu",
                "kernels/rmsnorm_backward.cu",
            ],
            base_dir=base_dir,
        )
    return _MOD


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps):
        mod = _load_module()

        x_c = x.contiguous()
        gamma_c = gamma.contiguous()

        y, inv_rms = mod.rmsnorm_forward(x_c, gamma_c, float(eps))
        ctx.save_for_backward(x_c, gamma_c, inv_rms)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        mod = _load_module()
        x, gamma, inv_rms = ctx.saved_tensors

        grad_y_c = grad_y.contiguous()
        grad_x, grad_gamma = mod.rmsnorm_backward(grad_y_c, x, gamma, inv_rms)

        return grad_x, grad_gamma, None


def candidate(inputs, params):
    x = inputs["X"]
    gamma = inputs["gamma"]
    eps = float(params.get("eps", 1e-5))
    return _RMSNormFunction.apply(x, gamma, eps)


def forward_only(inputs, params):
    mod = _load_module()

    x = inputs["X"].contiguous()
    gamma = inputs["gamma"].contiguous()
    eps = float(params.get("eps", 1e-5))

    y, _ = mod.rmsnorm_forward(x, gamma, eps)
    return y
