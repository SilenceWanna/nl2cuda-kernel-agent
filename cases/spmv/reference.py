"""Vectorized PyTorch reference for CSR SpMM/SpMV."""

import torch

from cases.spmv import config


def reference_forward(inputs, params):
    del params
    row_ptr = inputs["row_ptr"]
    col_idx = inputs["col_idx"]
    vals = inputs["vals"]
    X = inputs["X"]

    row_counts = row_ptr[1:] - row_ptr[:-1]
    row_idx = torch.repeat_interleave(
        torch.arange(row_counts.shape[0], device=row_ptr.device), row_counts
    )
    contributions = vals.unsqueeze(1) * X.index_select(0, col_idx)
    output = torch.zeros(
        (row_counts.shape[0], X.shape[1]), dtype=X.dtype, device=X.device
    )
    return output.index_add_(0, row_idx, contributions)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    nnz = config.M * config.NNZ_PER_ROW
    row_ptr = torch.arange(
        config.M + 1, dtype=torch.int64, device=device
    ) * config.NNZ_PER_ROW
    col_idx = torch.randint(
        0,
        config.K,
        (nnz,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    vals = torch.randn(nnz, dtype=dtype, device=device, generator=generator)
    X = torch.randn(
        config.K, config.D, dtype=dtype, device=device, generator=generator
    )
    if requires_grad:
        vals.requires_grad_(True)
        X.requires_grad_(True)
    return {"row_ptr": row_ptr, "col_idx": col_idx, "vals": vals, "X": X}
