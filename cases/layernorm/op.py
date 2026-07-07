"""LayerNorm kernel 的候选实现封装（dict 接口，符合 framework 候选契约）。

candidate(inputs, params) -> Y，对 inputs["X"]/["gamma"]/["beta"] 提供梯度。
kernel 源在 cases/layernorm/kernels/，由 framework.loader 即时编译。float32-only。
"""

import os
import functools

import torch

from framework.loader import load_kernel

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _fwd_module():
    return load_kernel("layernorm_forward", ["layernorm_forward.cu"], base_dir=_KDIR)


@functools.lru_cache(maxsize=1)
def _bwd_module():
    return load_kernel("layernorm_backward", ["layernorm_backward.cu"], base_dir=_KDIR)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, eps):
        Y = _fwd_module().layernorm_forward(X, gamma, beta, float(eps))
        ctx.save_for_backward(X, gamma)
        ctx.eps = float(eps)
        return Y

    @staticmethod
    def backward(ctx, G):
        X, gamma = ctx.saved_tensors
        dX, dgamma, dbeta = _bwd_module().layernorm_backward(
            X, G.contiguous(), gamma, ctx.eps)
        # 对应 forward 的 (X, gamma, beta, eps)
        return dX, dgamma, dbeta, None


def candidate(inputs, params):
    """framework 候选契约：inputs={"X","gamma","beta"}, params={"eps"} -> Y。"""
    return LayerNormFunction.apply(inputs["X"], inputs["gamma"], inputs["beta"],
                                   params["eps"])


def forward_only(inputs, params):
    """仅前向（no autograd），供调试。"""
    return _fwd_module().layernorm_forward(
        inputs["X"], inputs["gamma"], inputs["beta"], float(params["eps"]))
