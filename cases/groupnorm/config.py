"""Shape and scalar configuration for the GroupNorm case."""

import os


N = int(os.environ.get("GN_N", "64"))
C = int(os.environ.get("GN_C", "128"))
H = int(os.environ.get("GN_H", "56"))
W = int(os.environ.get("GN_W", "56"))
GROUPS = int(os.environ.get("GN_GROUPS", "32"))
EPS = 1e-5
DTYPE = "float32"

if C <= 0 or GROUPS <= 0 or C % GROUPS != 0:
    raise ValueError("GN_C must be positive and divisible by GN_GROUPS")
if N <= 0 or H <= 0 or W <= 0:
    raise ValueError("GN_N, GN_H, and GN_W must be positive")
