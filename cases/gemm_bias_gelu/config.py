import os


M = int(os.environ.get("GBG_M", "8192"))
K = int(os.environ.get("GBG_K", "256"))
N = int(os.environ.get("GBG_N", "1024"))
DTYPE = "float32"

