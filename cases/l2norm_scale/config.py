"""l2norm_scale case 的算法特定配置（shape / 标量参数）。

形状支持 env 覆盖，便于短核放大重测：L2N_N / L2N_D。
"""

import os

N = int(os.environ.get("L2N_N", "16384"))
D = int(os.environ.get("L2N_D", "1024"))
EPS = float(os.environ.get("L2N_EPS", "1e-6"))
DTYPE = "float32"
