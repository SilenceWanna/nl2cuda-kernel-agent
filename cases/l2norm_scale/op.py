"""l2norm_scale kernel 的候选实现封装（dict 接口，符合 framework 候选契约）。

candidate(inputs, params) -> Y，对 inputs["X"]/["g"] 提供梯度。
前向缓存 norm 供反向复用（避免反向重算 L2 模长）。float32-only。
"""

import os
import functools

import torch

from framework.loader import load_kernel

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _ext():
    return load_kernel("l2norm_scale_ext", ["l2norm_scale.cu"], base_dir=_KDIR)


class L2NormScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, g, eps):
        Y, norm = _ext().l2n_forward(X.contiguous(), g.contiguous(), float(eps))
        ctx.save_for_backward(X, g, norm)
        return Y

    @staticmethod
    def backward(ctx, G):
        X, g, norm = ctx.saved_tensors
        dX, dg = _ext().l2n_backward(X, g, G.contiguous(), norm)
        # 对应 forward 的 (X, g, eps)
        return dX, dg, None


def candidate(inputs, params):
    return L2NormScaleFunction.apply(inputs["X"], inputs["g"], params["eps"])


def forward_only(inputs, params):
    Y, _ = _ext().l2n_forward(
        inputs["X"].contiguous(), inputs["g"].contiguous(), float(params["eps"]))
    return Y
