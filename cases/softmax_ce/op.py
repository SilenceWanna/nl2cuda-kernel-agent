import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "softmax_ce_ext",
        ["softmax_ce_forward.cu", "softmax_ce_backward.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _SoftmaxCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        logits = logits.contiguous()
        labels = labels.contiguous()

        if ctx.needs_input_grad[0]:
            loss, probs = _extension().softmax_ce_forward(logits, labels)
            ctx.save_for_backward(probs, labels)
        else:
            loss = _extension().softmax_ce_forward_only(logits, labels)
            ctx.save_for_backward()
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        if not saved:
            return None, None
        probs, labels = saved
        grad_logits = _extension().softmax_ce_backward(
            grad_out.contiguous(),
            probs,
            labels,
        )
        return grad_logits, None


def candidate(inputs, params):
    return _SoftmaxCEFunction.apply(inputs["logits"], inputs["labels"])


def forward_only(inputs, params):
    return _extension().softmax_ce_forward_only(
        inputs["logits"].contiguous(),
        inputs["labels"].contiguous(),
    )
