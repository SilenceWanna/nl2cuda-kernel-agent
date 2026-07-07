"""全局计时/容差协议（算法无关，跨所有 case 统一，保证公平对比）。

对应工作目标第五节计时协议 + 第三节正确性判据的通用部分。
算法特定的 shape / 标量参数不在这里——它们在各 case 的 config 里。

⚠️ 本文件属评测基座，对 agent 只读，不得修改/绕过。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    # 正确性判据（通用）
    atol: float = 1e-2
    rtol: float = 1e-2
    seeds: tuple = (0, 1, 2, 3, 4)   # ≥5 组独立随机输入

    # 计时协议（防测量偏差）
    warmup: int = 10                 # ≥10
    iters: int = 100                 # 正式测量 ≥100，取几何均值
    repeats: int = 3                 # 重跑次数，算 CV
    cv_threshold: float = 0.05       # CV > 5% 作废重测
    max_retries: int = 4             # CV 超阈值时自动重测的最大次数（防共享环境噪声）

    # 达标阈值：自定义 kernel 相对 torch.compile 前反向均需 ≥1.05x
    speedup_target: float = 1.05


PROTOCOL = Protocol()


def dtype_of(name):
    import torch
    return {"float32": torch.float32, "float64": torch.float64,
            "float16": torch.float16}[name]
