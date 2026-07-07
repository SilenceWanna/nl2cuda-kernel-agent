"""RBF 高斯核矩阵的 PyTorch 参考实现（正确性金标准）。

结构：
    X:[N, D], Y:[M, D]
    Dist[i,j] = ||x_i - y_j||^2
    K[i,j]    = exp(-gamma * Dist[i,j])

关键（赢面前提）：前向用**自然广播形式** (x[:,None,:] - y[None,:,:]).pow(2).sum(-1)
表达，绝不写成 ||x||^2 + ||y||^2 - 2 X Y^T 的 GEMM 分解——后者会让 torch.compile
走 cuBLAS，几乎无法被手写 kernel 打败。广播形式是自然语言描述最直接的翻译，正当。

反向由 autograd 自动求得，因此本实现同时是前向和反向的金标准。

数学参考（供手写 CUDA kernel 对照，不用于本文件）：
    设 upstream 梯度为 G[i,j] = dL/dK[i,j]，K[i,j] = exp(-gamma * dist_ij)。
    令 S[i,j] = G[i,j] * K[i,j] * (-gamma)。
    则 d(dist_ij)/dx_i = 2 (x_i - y_j),  d(dist_ij)/dy_j = -2 (x_i - y_j)。
    dX[i] = sum_j S[i,j] * 2 (x_i - y_j)
    dY[j] = sum_i S[i,j] * (-2)(x_i - y_j)
"""

import torch


def rbf_kernel_reference(X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
    """RBF 核矩阵前向（广播形式）。X:[N,D], Y:[M,D] -> K:[N,M]。

    反向依赖 autograd：X/Y 需 requires_grad=True 时，K.backward() 会自动填充 X.grad/Y.grad。
    """
    # [N,1,D] - [1,M,D] -> [N,M,D]，物化 N*M*D 中间张量（正是手写 kernel 要避免的）
    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    dist_sq = diff.pow(2).sum(dim=-1)          # [N,M]
    return torch.exp(-gamma * dist_sq)


def make_inputs(N: int, M: int, D: int, dtype: torch.dtype, device: str,
                seed: int, requires_grad: bool = False):
    """按种子生成一组独立随机输入 (X, Y)。"""
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(N, D, dtype=dtype, device=device, generator=g)
    Y = torch.randn(M, D, dtype=dtype, device=device, generator=g)
    if requires_grad:
        X.requires_grad_(True)
        Y.requires_grad_(True)
    return X, Y


if __name__ == "__main__":
    # 冒烟测试：小规模跑通前向 + 反向（autograd），确认无语法/形状错误。
    # 在无 GPU 的本地会用 CPU；有 GPU 时自动用 cuda。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    N, M, D, gamma = 8, 6, 4, 1.0 / 4.0

    X, Y = make_inputs(N, M, D, dtype, device, seed=0, requires_grad=True)
    K = rbf_kernel_reference(X, Y, gamma)
    print(f"device={device} K.shape={tuple(K.shape)} "
          f"K.min={K.min().item():.4f} K.max={K.max().item():.4f}")

    # 反向冒烟：随机 upstream 梯度
    G = torch.randn_like(K)
    K.backward(G)
    print(f"dX.shape={tuple(X.grad.shape)} dY.shape={tuple(Y.grad.shape)} "
          f"dX.norm={X.grad.norm().item():.4f} dY.norm={Y.grad.norm().item():.4f}")
    print("reference forward+backward OK")
