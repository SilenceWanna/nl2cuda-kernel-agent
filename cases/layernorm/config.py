import os

B = int(os.environ.get("LAYERNORM_B", os.environ.get("LN_B", "4096")))
D = int(os.environ.get("LAYERNORM_D", os.environ.get("LN_D", "1024")))
EPS = 1.0e-5
DTYPE = "float32"
