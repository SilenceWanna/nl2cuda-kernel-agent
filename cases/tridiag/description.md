# Batched Tridiagonal Solve

Given contiguous FP32 tensors `lower`, `diag`, `upper`, and `rhs`, each with
shape `[B, N]`, solve one tridiagonal system per batch row:

\[
lower_{b,i} x_{b,i-1} + diag_{b,i} x_{b,i}
+ upper_{b,i} x_{b,i+1} = rhs_{b,i}.
\]

The boundary entries satisfy `lower[:, 0] = 0` and `upper[:, N-1] = 0`.
The default benchmark uses `B = 8192`, `N = 512`, and strictly diagonally
dominant matrices. The output `x` has shape `[B, N]`.

The CUDA candidate uses the Thomas algorithm. Its forward pass stores the LU
multipliers and pivots. For upstream gradient `g`, the backward pass solves

\[
A^T \lambda = g
\]

using the saved factors, then returns gradients for all four inputs:

\[
d(rhs)_i = \lambda_i, \qquad d(diag)_i = -\lambda_i x_i,
\]

\[
d(lower)_i = -\lambda_i x_{i-1}, \qquad
d(upper)_i = -\lambda_i x_{i+1},
\]

with zero gradients at the unused boundary entries. The PyTorch reference uses
nine explicitly expanded parallel cyclic-reduction stages, providing a
vectorized `torch.compile` baseline with the same solve semantics.
