"""Autograd wrapper for the custom CUDA LayerNorm implementation."""

import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_kernel(
            name="layernorm_cuda_ext",
            sources=["layernorm_forward.cu", "layernorm_backward.cu"],
            base_dir=_KERNEL_DIR,
            verbose=True,
        )
    return _MODULE


class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        mod = _load_module()

        x_c = x.contiguous()
        gamma_c = gamma.contiguous()
        beta_c = beta.contiguous()

        y, mean, inv_std = mod.layernorm_forward(x_c, gamma_c, beta_c, float(eps))
        ctx.save_for_backward(x_c, gamma_c, mean, inv_std)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, gamma, mean, inv_std = ctx.saved_tensors
        mod = _load_module()

        grad_y_c = grad_y.contiguous()
        grad_x, grad_gamma, grad_beta = mod.layernorm_backward(
            grad_y_c, x, gamma, mean, inv_std
        )
        return grad_x, grad_gamma, grad_beta, None


def candidate(inputs, params):
    """Candidate implementation entrypoint required by framework.

    Args:
      inputs: {"X": [B,D], "gamma": [D], "beta": [D]}
      params: {"eps": float}
    """
    eps = float(params.get("eps", 1.0e-5))
    return _LayerNormFunction.apply(inputs["X"], inputs["gamma"], inputs["beta"], eps)


def forward_only(inputs, params):
    """Convenience forward-only entrypoint."""
    return candidate(inputs, params)
