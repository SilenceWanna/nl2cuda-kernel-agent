import os

B = int(os.environ.get("RMSNORM_B", "4096"))
D = int(os.environ.get("RMSNORM_D", "1024"))
EPS = float(os.environ.get("RMSNORM_EPS", "1e-5"))
