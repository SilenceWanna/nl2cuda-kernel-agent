import os


B = int(os.environ.get("CONV1D_B", "64"))
C = int(os.environ.get("CONV1D_C", "256"))
T = int(os.environ.get("CONV1D_T", "1024"))
K = 4
DTYPE = "float32"
