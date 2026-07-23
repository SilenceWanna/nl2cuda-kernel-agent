"""Shape and dtype settings for the tanh-GeGLU case."""

import os


# The default has 32,768 token positions and is large enough to avoid a
# launch-overhead-dominated benchmark. Each dimension can be overridden.
B = int(os.environ.get("GEGLU_B", "16"))
T = int(os.environ.get("GEGLU_T", "2048"))
H = int(os.environ.get("GEGLU_H", "4096"))
DTYPE = "float32"
