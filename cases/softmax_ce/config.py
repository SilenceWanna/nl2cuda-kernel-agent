NAME = "softmax_ce"

DEFAULT_BATCH = 4096
DEFAULT_CLASSES = 1024

DESCRIPTION = """
Softmax cross entropy over a 2D logits tensor.

Inputs:
- logits: Tensor[N, C], floating point CUDA tensor
- target: Tensor[N], int64 class indices

Output:
- scalar mean cross entropy loss

Reference implementation must use primitive PyTorch operators:
logsumexp + gather.
"""
