"""RBF kernel 的候选实现封装（dict 接口，符合 framework 的候选契约）。

candidate(inputs, params) -> K，对 inputs["X"]/inputs["Y"] 提供梯度。
kernel 源在 cases/rbf/kernels/，由 framework.loader 即时编译。float32-only。
"""

import os
import functools

import torch

from framework.loader import load_kernel

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _fwd_module():
    return load_kernel("rbf_forward", ["rbf_forward.cu"], base_dir=_KDIR)


@functools.lru_cache(maxsize=1)
def _bwd_module():
    return load_kernel("rbf_backward", ["rbf_backward.cu"], base_dir=_KDIR)


class RBFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y, gamma):
        K = _fwd_module().rbf_forward(X, Y, float(gamma))
        ctx.save_for_backward(X, Y, K)   # 缓存 K 供反向复用，免重算 dist/exp
        ctx.gamma = float(gamma)
        return K

    @staticmethod
    def backward(ctx, G):
        X, Y, K = ctx.saved_tensors
        dX, dY = _bwd_module().rbf_backward(X, Y, G.contiguous(), K, ctx.gamma)
        return dX, dY, None


def candidate(inputs, params):
    """framework 候选契约：inputs={"X","Y"}, params={"gamma"} -> K。"""
    return RBFFunction.apply(inputs["X"], inputs["Y"], params["gamma"])


def forward_only(inputs, params):
    """仅前向（no autograd），供前向单独对拍/调试。"""
    return _fwd_module().rbf_forward(inputs["X"], inputs["Y"], float(params["gamma"]))
