import os

M = int(os.environ.get("GEMM_BIAS_GELU_M", "128"))
K = int(os.environ.get("GEMM_BIAS_GELU_K", "128"))
N = int(os.environ.get("GEMM_BIAS_GELU_N", "128"))
