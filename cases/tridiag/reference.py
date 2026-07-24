"""Vectorized PCR correctness and torch.compile reference."""

import torch

from cases.tridiag import config


def _pcr_step(lower, diag, upper, rhs, index, stride):
    left_diag = torch.roll(diag, stride, dims=1)
    right_diag = torch.roll(diag, -stride, dims=1)
    alpha = torch.where(index >= stride, -lower / left_diag, 0.0)
    beta = torch.where(index + stride < diag.shape[1], -upper / right_diag, 0.0)

    lower_next = alpha * torch.roll(lower, stride, dims=1)
    diag_next = (
        diag
        + alpha * torch.roll(upper, stride, dims=1)
        + beta * torch.roll(lower, -stride, dims=1)
    )
    upper_next = beta * torch.roll(upper, -stride, dims=1)
    rhs_next = (
        rhs
        + alpha * torch.roll(rhs, stride, dims=1)
        + beta * torch.roll(rhs, -stride, dims=1)
    )
    return lower_next, diag_next, upper_next, rhs_next


def reference_forward(inputs, params):
    lower = inputs["lower"]
    diag = inputs["diag"]
    upper = inputs["upper"]
    rhs = inputs["rhs"]
    index = torch.arange(diag.shape[1], device=diag.device)

    state = _pcr_step(lower, diag, upper, rhs, index, 1)
    state = _pcr_step(*state, index, 2)
    state = _pcr_step(*state, index, 4)
    state = _pcr_step(*state, index, 8)
    state = _pcr_step(*state, index, 16)
    state = _pcr_step(*state, index, 32)
    state = _pcr_step(*state, index, 64)
    state = _pcr_step(*state, index, 128)
    state = _pcr_step(*state, index, 256)
    return state[3] / state[1]


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (config.B, config.N)
    lower = 0.2 * torch.randn(shape, dtype=dtype, device=device, generator=generator)
    upper = 0.2 * torch.randn(shape, dtype=dtype, device=device, generator=generator)
    rhs = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0
    diag = 1.0 + lower.abs() + upper.abs()

    if requires_grad:
        lower.requires_grad_(True)
        diag.requires_grad_(True)
        upper.requires_grad_(True)
        rhs.requires_grad_(True)
    return {"lower": lower, "diag": diag, "upper": upper, "rhs": rhs}
