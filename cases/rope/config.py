"""RoPE case shape and scalar configuration."""

import os


B = int(os.environ.get("ROPE_B", "8"))
S = int(os.environ.get("ROPE_S", "1024"))
H = int(os.environ.get("ROPE_H", "16"))
D = int(os.environ.get("ROPE_D", "64"))
BASE = 10000.0
DTYPE = "float32"

if min(B, S, H, D) <= 0:
    raise ValueError("ROPE_B, ROPE_S, ROPE_H, and ROPE_D must be positive")
if D % 2 != 0:
    raise ValueError("ROPE_D must be even")
