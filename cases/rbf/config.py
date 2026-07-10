"""RBF case shape and scalar parameter configuration."""

import os


N = int(os.environ.get("RBF_SIZE", "2048"))
M = N
D = 64

GAMMA = 1.0 / D
DTYPE = "float32"
