"""Softmax 交叉熵 kernel 的候选实现封装（dict 接口，符合 framework 候选契约）。

candidate(inputs, params) -> 标量 loss，对 inputs["logits"] 提供梯度（labels 整型不求）。
kernel 源在 cases/softmax_ce/kernels/，float32-only（labels int64）。
"""

import os
import functools

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
    def backward(ctx, gout):
        logits, labels = ctx.saved_tensors
        # gout 是标量上游梯度；传给 kernel（保证 contiguous 的 1 元素张量）
        g = gout.reshape(1).contiguous()
        dlogits = _bwd_module().softmax_ce_backward(logits, labels, g)
        return dlogits, None      # 对应 (logits, labels)；labels 不求梯度


def candidate(inputs, params):
    """framework 候选契约：inputs={"logits","labels"} -> 标量 loss。"""
    return SoftmaxCEFunction.apply(inputs["logits"], inputs["labels"])
