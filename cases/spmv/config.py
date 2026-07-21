"""Shape and dtype configuration for CSR SpMM/SpMV."""

import os


M = int(os.environ.get("SPMV_M", "4096"))
K = int(os.environ.get("SPMV_K", "4096"))
D = int(os.environ.get("SPMV_D", "64"))
NNZ_PER_ROW = int(os.environ.get("SPMV_NNZ_PER_ROW", "64"))
DTYPE = "float32"
