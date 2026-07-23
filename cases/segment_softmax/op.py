"""Autograd wrapper for the segment-softmax CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _extension():
    return load_kernel(
        "segment_softmax_cuda",
        ["segment_softmax.cu"],
        base_dir=_KERNEL_DIR,
        verbose=False,
    )


class _SegmentSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, segment_ids, num_segments):
        segment_ids = segment_ids.contiguous()
        output = _extension().segment_softmax_forward(
            values.contiguous(), segment_ids, int(num_segments)
        )
        ctx.save_for_backward(output, segment_ids)
        ctx.num_segments = int(num_segments)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, segment_ids = ctx.saved_tensors
        grad_values = _extension().segment_softmax_backward(
            output, grad_output.contiguous(), segment_ids, ctx.num_segments
        )
        return grad_values, None, None


def segment_softmax(values, segment_ids, num_segments):
    """Apply softmax over rows that share the same segment id."""
    return _SegmentSoftmaxFunction.apply(values, segment_ids, int(num_segments))


scatter_softmax = segment_softmax


def candidate(inputs, params):
    return segment_softmax(
        inputs["values"], inputs["segment_ids"], params["num_segments"]
    )


def forward_only(inputs, params):
    return _extension().segment_softmax_forward(
        inputs["values"].contiguous(),
        inputs["segment_ids"].contiguous(),
        int(params["num_segments"]),
    )
