"""Autograd wrapper for the temperature softmax CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _module():
    return load_kernel(
        "temperature_softmax_cuda",
        ["temperature_softmax.cu"],
        base_dir=_KERNEL_DIR,
    )


class TemperatureSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, temperature):
        probabilities = _module().temperature_softmax_forward(
            scores, float(temperature)
        )
        ctx.save_for_backward(probabilities)
        ctx.inv_temperature = 1.0 / float(temperature)
        return probabilities

    @staticmethod
    def backward(ctx, grad_output):
        (probabilities,) = ctx.saved_tensors
        grad_scores = _module().temperature_softmax_backward(
            probabilities,
            grad_output.contiguous(),
            ctx.inv_temperature,
        )
        return grad_scores, None


def candidate(inputs, params):
    return TemperatureSoftmaxFunction.apply(
        inputs["scores"], params["temperature"]
    )


def forward_only(inputs, params):
    return _module().temperature_softmax_forward(
        inputs["scores"], float(params["temperature"])
    )
