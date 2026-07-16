"""Autograd wrapper for custom causal attention CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "causal_attn_ext",
        ["causal_attn.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _CausalAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        out, probs = _extension().causal_attn_forward(Q, K, V)
        ctx.save_for_backward(Q, K, V, probs)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, probs = ctx.saved_tensors
        grad_Q, grad_K, grad_V = _extension().causal_attn_backward(
            grad_out.contiguous(), Q, K, V, probs
        )
        return grad_Q, grad_K, grad_V


def candidate(inputs, params):
    return _CausalAttentionFunction.apply(inputs["Q"], inputs["K"], inputs["V"])


def forward_only(inputs, params):
    out, _ = _extension().causal_attn_forward(
        inputs["Q"].contiguous(),
        inputs["K"].contiguous(),
        inputs["V"].contiguous(),
    )
    return out
