"""Shape and dtype configuration for segment softmax."""

import os


N = int(os.environ.get("SEGMENT_SOFTMAX_N", "1048576"))
FEATURES = int(os.environ.get("SEGMENT_SOFTMAX_FEATURES", "8"))
NUM_SEGMENTS = int(os.environ.get("SEGMENT_SOFTMAX_S", "65536"))
DTYPE = "float32"
