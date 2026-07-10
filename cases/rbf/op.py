"""Autograd wrapper for the RBF CUDA kernels."""

import os

import torch

from framework.loader import load_kernel


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_DIR = os.path.join(_THIS_DIR, "kernels")
_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        _EXT = load_kernel(
            name="rbf_kernel",
            sources=["rbf_forward.cu", "rbf_backward.cu"],
            base_dir=_KERNEL_DIR,
            verbose=False,
        )
    return _EXT


def forward_only(inputs, params):
    """Forward-only candidate entry point."""
    X = inputs["X"].contiguous()
    Y = inputs["Y"].contiguous()
    gamma = float(params["gamma"])
    return _load_ext().rbf_forward(X, Y, gamma)


class _RBFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y, gamma):
        Xc = X.contiguous()
        Yc = Y.contiguous()
        gamma_f = float(gamma)
        K = _load_ext().rbf_forward(Xc, Yc, gamma_f)
        ctx.save_for_backward(Xc, Yc, K)
        ctx.gamma = gamma_f
        return K

    @staticmethod
    def backward(ctx, grad_out):
        X, Y, K = ctx.saved_tensors
        grad_out_c = grad_out.contiguous()
        dX, dY = _load_ext().rbf_backward(grad_out_c, K, X, Y, ctx.gamma)
        return dX, dY, None


def candidate(inputs, params):
    """Candidate implementation with custom CUDA forward and backward."""
    return _RBFFunction.apply(inputs["X"], inputs["Y"], float(params["gamma"]))
