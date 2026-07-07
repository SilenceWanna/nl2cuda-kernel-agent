"""LayerNorm case 的算法特定配置（shape / 标量参数）。

通用计时/容差协议在 framework/protocol.py，跨 case 统一。
形状可切换：env LN_B / LN_D 覆盖默认。
"""

import os

B = int(os.environ.get("LN_B", "4096"))   # 行数（batch*tokens）
D = int(os.environ.get("LN_D", "1024"))   # 归一化维度
EPS = 1e-5
DTYPE = "float32"
