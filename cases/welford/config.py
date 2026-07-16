"""Shape and scalar configuration for the Welford LayerNorm case."""

import os


B = int(os.environ.get("WELFORD_B", "32"))
N = int(os.environ.get("WELFORD_N", "128"))
D = int(os.environ.get("WELFORD_D", "1024"))
EPS = 1.0e-5
DTYPE = "float32"
