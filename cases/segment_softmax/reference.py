"""Vectorized PyTorch reference for an unordered segment softmax."""

import math

import torch

from cases.segment_softmax import config


def reference_forward(inputs, params):
    values = inputs["values"]
    segment_ids = inputs["segment_ids"]
    num_segments = params["num_segments"]

    features = math.prod(values.shape[1:]) if values.dim() > 1 else 1
    flat_values = values.reshape(values.size(0), features)
    index = segment_ids[:, None].expand(-1, features)

    maxima = torch.full(
        (num_segments, features),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    maxima.scatter_reduce_(0, index, flat_values, reduce="amax", include_self=True)
    exponentials = torch.exp(flat_values - maxima.index_select(0, segment_ids))

    sums = torch.zeros_like(maxima)
    sums.scatter_add_(0, index, exponentials)
    return (exponentials / sums.index_select(0, segment_ids)).reshape_as(values)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    values = torch.randn(
        config.N,
        config.FEATURES,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    segment_ids = torch.randint(
        0,
        config.NUM_SEGMENTS,
        (config.N,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    if requires_grad:
        values.requires_grad_(True)
    return {"values": values, "segment_ids": segment_ids}
