"""RBF case 的算法特定配置（shape / 标量参数）。

通用计时/容差协议（warmup/iters/cv/target/atol/rtol/seeds）不在这里——
它们在 framework/protocol.py，跨 case 统一。

形状可切换：默认 N=M=2048（T4 安全），env RBF_SIZE=4096 恢复大形状（需 ≥24GB 卡）。
"""

import os

_SIZE = int(os.environ.get("RBF_SIZE", "2048"))

N = _SIZE
M = _SIZE
D = 64
GAMMA = 1.0 / 64.0     # K = exp(-gamma * dist^2)；取 1/D 使 exp 输入落在合理数值区间
DTYPE = "float32"      # 验收精度（禁止降精度换速度）
