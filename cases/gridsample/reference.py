"""Vectorized PyTorch reference for bilinear grid sampling."""

import torch

from cases.gridsample import config


def _pixel_coordinates(coord, size):
    coord = torch.where(torch.isnan(coord), torch.full_like(coord, -1.0), coord)
    return ((coord + 1.0) * size - 1.0) * 0.5


def reference_forward(inputs, params):
    X = inputs["X"]
    grid = inputs["grid"].detach()
    N, C, H, W = X.shape
    _, OH, OW, two = grid.shape
    if two != 2:
        raise ValueError("grid must have shape [N, OH, OW, 2]")

    ix = _pixel_coordinates(grid[..., 0], W)
    iy = _pixel_coordinates(grid[..., 1], H)
    x0 = torch.floor(ix).to(torch.long)
    y0 = torch.floor(iy).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    wx1 = ix - x0.to(ix.dtype)
    wy1 = iy - y0.to(iy.dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    x0_valid = (x0 >= 0) & (x0 < W)
    x1_valid = (x1 >= 0) & (x1 < W)
    y0_valid = (y0 >= 0) & (y0 < H)
    y1_valid = (y1 >= 0) & (y1 < H)
    x0c = x0.clamp(0, W - 1)
    x1c = x1.clamp(0, W - 1)
    y0c = y0.clamp(0, H - 1)
    y1c = y1.clamp(0, H - 1)

    n_index = torch.arange(N, device=X.device)[:, None, None].expand(N, OH, OW)
    image = X.permute(0, 2, 3, 1)
    v00 = image[n_index, y0c, x0c]
    v01 = image[n_index, y0c, x1c]
    v10 = image[n_index, y1c, x0c]
    v11 = image[n_index, y1c, x1c]
    v00 = v00 * (x0_valid & y0_valid)[..., None]
    v01 = v01 * (x1_valid & y0_valid)[..., None]
    v10 = v10 * (x0_valid & y1_valid)[..., None]
    v11 = v11 * (x1_valid & y1_valid)[..., None]
    output = (v00 * (wx0 * wy0)[..., None] +
              v01 * (wx1 * wy0)[..., None] +
              v10 * (wx0 * wy1)[..., None] +
              v11 * (wx1 * wy1)[..., None])
    return output.permute(0, 3, 1, 2).contiguous()


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(config.N, config.C, config.H, config.W,
                    dtype=dtype, device=device, generator=generator)
    grid = torch.rand(config.N, config.OH, config.OW, 2,
                      dtype=dtype, device=device, generator=generator) * 2.4 - 1.2
    if requires_grad:
        X.requires_grad_(True)
    return {"X": X, "grid": grid}
