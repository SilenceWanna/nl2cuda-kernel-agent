"""Shape and dtype configuration for row-wise Top-K."""

import os


N = int(os.environ.get("TOPK_N", "16384"))
D = int(os.environ.get("TOPK_D", "1024"))
K = 8
DTYPE = "float32"

if N <= 0:
    raise ValueError("TOPK_N must be positive")
if D < K:
    raise ValueError(f"TOPK_D must be at least {K}")
