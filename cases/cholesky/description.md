# Cholesky decomposition

Factor one large, real symmetric positive-definite matrix
`A[N, N]` into a full-storage lower-triangular matrix `L[N, N]`:

`A = L @ L.T`, with `L[i, i] > 0` and `L[i, j] = 0` for `i < j`.

The default benchmark uses `N=4096` and float32 so that the single-matrix
cuSOLVER baseline is in the compute-dominated regime. `CHOLESKY_N` can override
the size for diagnostic scaling runs. The operator does not add diagonal jitter;
inputs are guaranteed SPD and are generated as `A = X @ X.T / N + I`.

The only differentiable input is `A`. It is interpreted as a full symmetric
variable, so its returned gradient is symmetric. Given upstream gradient `G`,
define

`Phi(X)[i,j] = X[i,j]` for `i > j`, `X[i,i] / 2` for `i = j`, and zero for
`i < j`. Then

`H = L^(-T) @ Phi(L.T @ G) @ L^(-1)`

and

`grad_A = (H + H.T) / 2`.

The reference uses `torch.linalg.cholesky_ex(..., check_errors=False)` as the
strong cuSOLVER-level `torch.compile` baseline. The candidate must execute its
own blocked path from the custom CUDA extension and may use CUDA vendor
primitives without calling Torch high-level tensor operations.
