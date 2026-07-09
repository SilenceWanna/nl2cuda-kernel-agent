"""Softmax cross-entropy PyTorch reference implementation (dict interface).

The reference intentionally uses primitive tensor ops (logsumexp + gather + mean),
so autograd provides the gold standard backward without falling back to a high-level
fused loss op.
"""

import torch

from cases.softmax_ce import config


def reference_forward(inputs, params):
    """inputs={"logits":[B,C], "labels":[B]} -> scalar mean cross-entropy loss."""
    logits = inputs["logits"]
    labels = inputs["labels"]
    log_z = torch.logsumexp(logits, dim=1)                  # [B]
    chosen = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    return (log_z - chosen).mean()


def make_inputs(seed, dtype, device, requires_grad=False):
    """Generate named inputs. Only logits can require gradients; labels stay int64."""
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(config.B, config.C, dtype=dtype, device=device, generator=g)
    labels = torch.randint(config.C, (config.B,), dtype=torch.int64, device=device, generator=g)
    if requires_grad:
        logits.requires_grad_(True)
    return {"logits": logits, "labels": labels}
