"""PyTorch reference for mean softmax cross-entropy.

The reference intentionally uses only primitive tensor operations.
"""

import torch

from cases.softmax_ce import config


def reference_forward(inputs, params):
    """inputs={"logits":[B,C], "labels":[B]} -> scalar mean loss."""
    del params
    logits = inputs["logits"]
    labels = inputs["labels"]

    log_z = torch.logsumexp(logits, dim=1)
    target_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    return (log_z - target_logits).mean()


def make_inputs(seed, dtype, device, requires_grad=False):
    """Create named inputs. Only logits may require gradients."""
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(config.B, config.C, dtype=dtype, device=device, generator=g)
    labels = torch.randint(
        low=0,
        high=config.C,
        size=(config.B,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    if requires_grad:
        logits.requires_grad_(True)
    return {"logits": logits, "labels": labels}
