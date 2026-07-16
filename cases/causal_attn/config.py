"""Causal single-head self-attention case configuration."""

import os

B = int(os.environ.get("CAUSAL_ATTN_B", "16"))
T = int(os.environ.get("CAUSAL_ATTN_T", "128"))
D = int(os.environ.get("CAUSAL_ATTN_D", "64"))
DTYPE = "float32"
