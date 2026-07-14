import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "scan_ext",
        ["scan.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _ScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _extension().scan_forward(x.contiguous())

    @staticmethod
    def backward(ctx, grad_out):
        return _extension().scan_backward(grad_out.contiguous())


def candidate(inputs, params):
    return _ScanFunction.apply(inputs["X"])


def forward_only(inputs, params):
    return _extension().scan_forward(inputs["X"].contiguous())

