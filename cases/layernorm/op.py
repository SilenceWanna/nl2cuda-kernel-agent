import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "layernorm_ext",
        ["layernorm_forward.cu", "layernorm_backward.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        eps = float(eps)

        out, mean, rstd = _extension().layernorm_forward(x, gamma, beta, eps)
        ctx.save_for_backward(x, gamma, mean, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, gamma, mean, rstd = ctx.saved_tensors
        grad_x, grad_gamma, grad_beta = _extension().layernorm_backward(
            grad_out.contiguous(),
            x,
            gamma,
            mean,
            rstd,
        )
        return grad_x, grad_gamma, grad_beta, None


def candidate(inputs, params):
    return _LayerNormFunction.apply(
        inputs["X"],
        inputs["gamma"],
        inputs["beta"],
        float(params["eps"]),
    )


def forward_only(inputs, params):
    out, _, _ = _extension().layernorm_forward(
        inputs["X"].contiguous(),
        inputs["gamma"].contiguous(),
        inputs["beta"].contiguous(),
        float(params["eps"]),
    )
    return out
