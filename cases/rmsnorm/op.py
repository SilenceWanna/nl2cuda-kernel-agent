import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_kernel(
            name="rmsnorm_cuda_ext",
            sources=["rmsnorm_forward.cu", "rmsnorm_backward.cu"],
            base_dir=_KERNEL_DIR,
            verbose=True,
        )
    return _MODULE


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps):
        mod = _load_module()

        x_c = x.contiguous()
        gamma_c = gamma.contiguous()

        y, inv_rms = mod.rmsnorm_forward(x_c, gamma_c, float(eps))
        ctx.save_for_backward(x_c, gamma_c, inv_rms)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, gamma, inv_rms = ctx.saved_tensors
        mod = _load_module()

        grad_y_c = grad_y.contiguous()
        grad_x, grad_gamma = mod.rmsnorm_backward(
            grad_y_c,
            x,
            gamma,
            inv_rms,
        )
        return grad_x, grad_gamma, None


def candidate(inputs, params):
    eps = float(params.get("eps", 1.0e-5))
    return _RMSNormFunction.apply(inputs["X"], inputs["gamma"], eps)


def forward_only(inputs, params):
    return candidate(inputs, params)
