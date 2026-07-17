# Scatter-add (segmented aggregation)

Inputs:

- `X`: float32 tensor with shape `[N, D]`; row `i` is a source vector.
- `idx`: int64 tensor with shape `[N]`; `idx[i]` is in `[0, S - 1]` and selects the destination segment for source row `i`.

Forward output `Y` has shape `[S, D]` and is defined by

`Y[s, d] = sum(X[i, d] for i where idx[i] == s)`.

Multiple source rows may accumulate into the same segment, and segments with no source rows produce zero rows. Compute gradients only for `X`; `idx` is an integer index and has no gradient. The backward relation is `dX[i, d] = dY[idx[i], d]`.
