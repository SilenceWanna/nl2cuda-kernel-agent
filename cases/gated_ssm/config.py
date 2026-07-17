import os


B = int(os.environ.get("GSSM_B", "64"))
T = int(os.environ.get("GSSM_T", "128"))
C = int(os.environ.get("GSSM_C", "128"))
DTYPE = "float32"
