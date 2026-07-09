"""Softmax cross-entropy candidate implementation (dict interface).

candidate(inputs, params) -> scalar loss, with gradients for inputs["logits"] only.
CUDA kernels live in cases/softmax_ce/kernels/ and are compiled through
framework.loader. float32-only; labels are int64 and non-differentiable.
"""

import functools
import os

import torch

from framework.loader import load_kernel

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _fwd_module():
    return load_kernel("softmax_ce_forward", ["softmax_ce_forward.cu"], base_dir=_KDIR)


@functools.lru_cache(maxsize=1)
def _bwd_module():
    return load_kernel("softmax_ce_backward", ["softmax_ce_backward.cu"], base_dir=_KDIR)


class SoftmaxCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        loss = _fwd_module().softmax_ce_forward(logits, labels)
        ctx.save_for_backward(logits, labels)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        logits, labels = ctx.saved_tensors
        dlogits = _bwd_module().softmax_ce_backward(logits, labels, grad_out.contiguous())
        return dlogits, None


def candidate(inputs, params):
    """framework candidate contract: inputs={"logits","labels"}, params={} -> scalar loss."""
    return SoftmaxCEFunction.apply(inputs["logits"], inputs["labels"])


def forward_only(inputs, params):
    """Forward-only helper for debugging/forward checks."""
    return _fwd_module().softmax_ce_forward(inputs["logits"], inputs["labels"])
