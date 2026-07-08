"""Softmax 交叉熵 case 的算法特定配置（shape / 参数）。

通用计时/容差协议在 framework/protocol.py。形状可切换：env SMCE_B / SMCE_C。
"""

import os

B = int(os.environ.get("SMCE_B", "8192"))   # batch（样本数）
C = int(os.environ.get("SMCE_C", "1024"))   # 类别数
DTYPE = "float32"                            # logits 精度（labels 恒为 int64）
