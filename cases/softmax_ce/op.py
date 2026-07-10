"""Autograd wrapper for the custom CUDA Softmax CE implementation."""

import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_kernel(
            name="softmax_ce_cuda_ext",
            sources=["softmax_ce_forward.cu", "softmax_ce_backward.cu"],
            base_dir=_KERNEL_DIR,
            verbose=True,
        )
    return _MODULE


class _SoftmaxCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        mod = _load_module()

        logits_c = logits.contiguous()
        labels_c = labels.contiguous()

        loss, logsumexp = mod.softmax_ce_forward(logits_c, labels_c)
        ctx.save_for_backward(logits_c, labels_c, logsumexp)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        logits, labels, logsumexp = ctx.saved_tensors
        mod = _load_module()

        grad_logits = mod.softmax_ce_backward(
            grad_loss.contiguous(), logits, labels, logsumexp
        )
        return grad_logits, None


def candidate(inputs, params):
    """Candidate implementation entrypoint required by framework.

    Args:
      inputs: {"logits": [B,C] float32, "labels": [B] int64}
      params: unused for this case
    """
    return _SoftmaxCEFunction.apply(inputs["logits"], inputs["labels"])


def forward_only(inputs, params):
    """Convenience forward-only entrypoint."""
    return candidate(inputs, params)
