"""Shapes and fixed semantics for the grid-sample case."""

import os


N = int(os.environ.get("GRIDSAMPLE_N", "4"))
C = int(os.environ.get("GRIDSAMPLE_C", "32"))
H = int(os.environ.get("GRIDSAMPLE_H", "64"))
W = int(os.environ.get("GRIDSAMPLE_W", "64"))
OH = int(os.environ.get("GRIDSAMPLE_OH", str(H)))
OW = int(os.environ.get("GRIDSAMPLE_OW", str(W)))
DTYPE = "float32"

if min(N, C, H, W, OH, OW) <= 0:
    raise ValueError("all grid-sample dimensions must be positive")
