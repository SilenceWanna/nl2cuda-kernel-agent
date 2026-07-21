# CSR SpMM/SpMV

稀疏矩阵 `A` 的形状为 `[M, K]`，采用 CSR 格式：`row_ptr[M+1]` 给出每行非零元在 `col_idx` 和 `vals` 中的起止偏移，`col_idx[nnz]` 保存列号，`vals[nnz]` 保存 fp32 非零值。稠密输入 `X` 的形状为 `[K, D]`，dtype 为 fp32。

前向输出 `Y[M, D]`：

`Y[m, d] = sum_{j=row_ptr[m]}^{row_ptr[m+1]-1} vals[j] * X[col_idx[j], d]`

对 `vals` 和 `X` 求梯度；整数结构张量 `row_ptr`、`col_idx` 不可导。默认规模 `M=K=4096`、`D=64`、每行 64 个非零元。
