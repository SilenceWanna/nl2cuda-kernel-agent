import os

import torch

from framework.loader import load_kernel
from cases.rbf import config


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        _EXT = load_kernel(
            name="rbf_kernel",
            sources=[
                "kernels/rbf_forward.cu",
                "kernels/rbf_backward.cu",
            ],
            base_dir=base_dir,
        )
    return _EXT


class _RBFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gamma):
        ext = _load_ext()

        x_contig = x.contiguous()
        y_contig = y.contiguous()

        k = ext.forward(
            x_contig,
            y_contig,
            float(gamma),
        )

        ctx.save_for_backward(x_contig, y_contig, k)
        ctx.gamma = float(gamma)
        return k

    @staticmethod
    def backward(ctx, grad_k):
        x, y, k = ctx.saved_tensors
        ext = _load_ext()

        d_x, d_y = ext.backward(
            grad_k.contiguous(),
            x,
            y,
            k,
            float(ctx.gamma),
        )

        return d_x, d_y, None


def candidate(inputs, params):
    gamma = params.get("gamma", config.GAMMA)
    return _RBFFunction.apply(
        inputs["X"],
        inputs["Y"],
        gamma,
    )


def forward_only(inputs, params):
    gamma = params.get("gamma", config.GAMMA)
    ext = _load_ext()
    return ext.forward(
        inputs["X"].contiguous(),
        inputs["Y"].contiguous(),
        float(gamma),
    )
