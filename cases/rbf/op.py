import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "rbf_ext",
        ["rbf_forward.cu", "rbf_backward.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _RBFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gamma):
        x = x.contiguous()
        y = y.contiguous()
        gamma = float(gamma)
        out = _extension().rbf_forward(x, y, gamma)
        ctx.save_for_backward(x, y, out)
        ctx.gamma = gamma
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, y, out = ctx.saved_tensors
        grad_x, grad_y = _extension().rbf_backward(
            grad_out.contiguous(), x, y, out, ctx.gamma
        )
        return grad_x, grad_y, None


def candidate(inputs, params):
    return _RBFFunction.apply(inputs["X"], inputs["Y"], float(params["gamma"]))


def forward_only(inputs, params):
    return _extension().rbf_forward(
        inputs["X"].contiguous(),
        inputs["Y"].contiguous(),
        float(params["gamma"]),
    )
