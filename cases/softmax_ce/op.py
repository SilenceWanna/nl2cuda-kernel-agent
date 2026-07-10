import functools
import os

import torch

from framework.loader import load_kernel


@functools.lru_cache(maxsize=1)
def _module():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return load_kernel(
        name="softmax_ce_kernel",
        sources=["kernels/softmax_ce.cu"],
        base_dir=base_dir,
        extra_cuda_cflags=["-O3"],
    )


class SoftmaxCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        logits_contig = logits.contiguous()
        labels_contig = labels.contiguous()

        loss, probs = _module().forward(logits_contig, labels_contig)
        ctx.save_for_backward(probs, labels_contig)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        probs, labels = ctx.saved_tensors
        grad_loss_contig = grad_loss.contiguous()
        grad_logits = _module().backward(probs, labels, grad_loss_contig)
        return grad_logits, None


def forward_only(inputs, params):
    logits = inputs["logits"]
    labels = inputs["labels"]
    return _module().forward(logits.contiguous(), labels.contiguous())[0]


def candidate(inputs, params):
    return SoftmaxCEFunction.apply(inputs["logits"], inputs["labels"])
