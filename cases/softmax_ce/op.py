import functools
from pathlib import Path

import torch

from framework.loader import load_kernel


_BASE_DIR = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def _fwd_module():
    return load_kernel(
        "softmax_ce_forward",
        sources=["kernels/softmax_ce_forward.cu"],
        base_dir=_BASE_DIR,
    )


@functools.lru_cache(maxsize=1)
def _bwd_module():
    return load_kernel(
        "softmax_ce_backward",
        sources=["kernels/softmax_ce_backward.cu"],
        base_dir=_BASE_DIR,
    )


class SoftmaxCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        logits = logits.contiguous()
        target = target.contiguous()

        loss = _fwd_module().softmax_ce_forward(logits, target)

        ctx.save_for_backward(logits, target)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, target = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_logits = _bwd_module().softmax_ce_backward(grad_output, logits, target)

        return grad_logits, None


def candidate(inputs, params):
    logits, target = inputs
    return SoftmaxCEFunction.apply(logits, target)


def forward_only(inputs, params):
    logits, target = inputs
    logits = logits.contiguous()
    target = target.contiguous()

    return _fwd_module().softmax_ce_forward(logits, target)
