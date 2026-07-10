import os

import torch

from framework.loader import load_kernel
from cases.layernorm import config


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        _EXT = load_kernel(
            name="layernorm_kernel",
            sources=[
                "kernels/layernorm_forward.cu",
                "kernels/layernorm_backward.cu",
            ],
            base_dir=base_dir,
        )
    return _EXT


class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        ext = _load_ext()

        x_contig = x.contiguous()
        gamma_contig = gamma.contiguous()
        beta_contig = beta.contiguous()

        y, mean, rstd = ext.forward(
            x_contig,
            gamma_contig,
            beta_contig,
            float(eps),
        )

        ctx.save_for_backward(x_contig, gamma_contig, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, gamma, mean, rstd = ctx.saved_tensors
        ext = _load_ext()

        d_x, d_gamma, d_beta = ext.backward(
            grad_y.contiguous(),
            x,
            gamma,
            mean,
            rstd,
        )

        return d_x, d_gamma, d_beta, None


def candidate(inputs, params):
    eps = params.get("eps", config.EPS)
    return _LayerNormFunction.apply(
        inputs["X"],
        inputs["gamma"],
        inputs["beta"],
        eps,
    )


def forward_only(inputs, params):
    eps = params.get("eps", config.EPS)
    ext = _load_ext()
    y, _, _ = ext.forward(
        inputs["X"].contiguous(),
        inputs["gamma"].contiguous(),
        inputs["beta"].contiguous(),
        float(eps),
    )
    return y
