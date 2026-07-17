"""Shape and dtype configuration for scatter-add."""

import os


N = int(os.environ.get("SCATTER_ADD_N", "262144"))
D = int(os.environ.get("SCATTER_ADD_D", "128"))
S = int(os.environ.get("SCATTER_ADD_S", "32768"))
DTYPE = "float32"
