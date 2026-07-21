"""Temperature softmax case configuration."""

import os


B = int(os.environ.get("TSM_B", "4096"))
D = int(os.environ.get("TSM_D", "1024"))
TEMPERATURE = 0.7
DTYPE = "float32"
