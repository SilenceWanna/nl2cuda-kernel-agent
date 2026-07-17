# Row-wise Top-K

Input `X` has shape `[N, D]` and dtype `float32`. For each row, select its
largest `k=8` values and return them in descending order as `Y` with shape
`[N, 8]`.

The backward pass computes the gradient with respect to `X`. If `X[i, p]`
was selected as `Y[i, j]`, then `dX[i, p] = dY[i, j]`; all unselected
positions receive zero. The custom kernel uses the smaller original column
index as an internal tie-break; as with `torch.topk`, callers should not rely
on a particular selected position when values are tied.
