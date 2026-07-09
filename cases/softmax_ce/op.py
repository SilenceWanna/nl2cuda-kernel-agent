"""Candidate implementation for softmax cross-entropy.

candidate(inputs, params) returns a scalar loss and supplies gradients for
inputs["logits"]. The CUDA extension is float32-only.
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
        loss, row_max, inv_sum = _fwd_module().softmax_ce_forward(logits, labels)
        ctx.save_for_backward(logits, labels, row_max, inv_sum)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        logits, labels, row_max, inv_sum = ctx.saved_tensors
        dlogits = _bwd_module().softmax_ce_backward(
            logits,
            labels,
            grad_loss.contiguous(),
            row_max,
            inv_sum,
        )
        return dlogits, None


def candidate(inputs, params):
    del params
    return SoftmaxCEFunction.apply(inputs["logits"], inputs["labels"])


def forward_only(inputs, params):
    del params
    return _fwd_module().softmax_ce_forward(inputs["logits"], inputs["labels"])[0]
