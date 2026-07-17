import os


N = int(os.environ.get("MAXPOOL_N", "32"))
C = int(os.environ.get("MAXPOOL_C", "64"))
H = int(os.environ.get("MAXPOOL_H", "64"))
W = int(os.environ.get("MAXPOOL_W", "64"))
DTYPE = "float32"

if H <= 0 or W <= 0 or H % 2 != 0 or W % 2 != 0:
    raise ValueError("MAXPOOL_H and MAXPOOL_W must be positive even integers")
