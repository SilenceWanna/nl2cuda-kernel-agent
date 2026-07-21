"""cosine_sim case 配置（成对余弦相似度矩阵）。规模支持 env 覆盖。"""

import os

N = int(os.environ.get("COS_N", "2048"))   # A 的行数
M = int(os.environ.get("COS_M", "2048"))   # B 的行数
D = int(os.environ.get("COS_D", "64"))     # 特征维
DTYPE = "float32"
