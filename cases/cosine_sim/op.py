"""cosine_sim kernel 候选封装（dict 接口）。candidate(inputs, params) -> S[N,M]，对 A/B 求梯度。"""

import os
import functools

import torch

from framework.loader import load_kernel

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _ext():
    return load_kernel("cosine_sim_ext", ["cosine_sim.cu"], base_dir=_KDIR)


class CosineSimFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, eps):
        S, Ah, Bh, invA, invB = _ext().cos_forward(
            A.contiguous(), B.contiguous(), float(eps))
        ctx.save_for_backward(Ah, Bh, invA, invB)
        ctx.shape = (A.size(0), B.size(0), A.size(1))
        return S

    @staticmethod
    def backward(ctx, dS):
        Ah, Bh, invA, invB = ctx.saved_tensors
        N, M, D = ctx.shape
        dA, dB = _ext().cos_backward(Ah, Bh, invA, invB, dS.contiguous(), N, M, D)
        return dA, dB, None


def candidate(inputs, params):
    return CosineSimFunction.apply(inputs["A"], inputs["B"], params["eps"])


def forward_only(inputs, params):
    S = _ext().cos_forward(
        inputs["A"].contiguous(), inputs["B"].contiguous(), float(params["eps"]))[0]
    return S
