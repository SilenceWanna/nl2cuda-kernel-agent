"""验收用例的精确规格（正确性金标准 + 计时的唯一形状来源）。

结构：成对距离 / RBF 高斯核矩阵（非 attention）。
    X:[N, D], Y:[M, D]
    Dist[i,j] = ||x_i - y_j||^2
    K[i,j]    = exp(-gamma * Dist[i,j])     # 前向输出 K:[N, M]
    反向：对 X、Y 求梯度 dX、dY

约定：
- 所有实现（参考 / kernel）与计时都从本文件取形状，避免不一致。
- 参考实现必须用"自然广播形式"表达，见 rbf_reference.py（保赢面，不走 GEMM 分解）。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RBFConfig:
    # 验收主用例形状
    N: int = 4096          # X 行数
    M: int = 4096          # Y 行数
    D: int = 64            # 特征维度
    gamma: float = 1.0 / 64.0   # RBF 带宽系数：K = exp(-gamma * dist^2)；取 1/D 使 exp 输入落在合理数值区间
    dtype: str = "float32"      # 验收精度（禁止降精度换速度）

    # 正确性判据（前反向都要过）
    atol: float = 1e-2
    rtol: float = 1e-2
    seeds: tuple = (0, 1, 2, 3, 4)   # ≥5 组独立随机输入

    # 小规模用例：用 double 精度 torch.autograd.gradcheck 复核反向
    grad_N: int = 8
    grad_M: int = 8
    grad_D: int = 4

    # 计时协议（防测量偏差）
    warmup: int = 10        # ≥10
    iters: int = 100        # 正式测量 ≥100，取几何均值
    repeats: int = 3        # 重跑次数，算 CV
    cv_threshold: float = 0.05   # CV > 5% 作废重测

    # 达标阈值：自定义 kernel 相对 torch.compile 前反向均需 ≥1.05x
    speedup_target: float = 1.05


CONFIG = RBFConfig()


def summary() -> str:
    c = CONFIG
    return (
        f"RBF kernel matrix | X:[{c.N},{c.D}] Y:[{c.M},{c.D}] -> K:[{c.N},{c.M}] "
        f"| gamma={c.gamma:.6g} dtype={c.dtype} "
        f"| tol(atol={c.atol},rtol={c.rtol}) seeds={c.seeds} "
        f"| bench(warmup={c.warmup},iters={c.iters},repeats={c.repeats},"
        f"cv<={c.cv_threshold},target>={c.speedup_target}x)"
    )


if __name__ == "__main__":
    print(summary())
