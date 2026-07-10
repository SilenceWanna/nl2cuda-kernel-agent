"""LayerNorm case configuration."""

import os

B = int(os.environ.get("LN_B", "4096"))  # 支持 env 覆盖：短核 case 计时前放大规模避免 CV 噪声
D = 1024
EPS = 1.0e-5
