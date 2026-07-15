import os


B = int(os.environ.get("LSSM_B", "512"))
T = int(os.environ.get("LSSM_T", "512"))
C = int(os.environ.get("LSSM_C", "128"))
A = 0.9
B_COEF = 1.0
DTYPE = "float32"
