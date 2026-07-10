"""PyTorch reference implementation for the Softmax CE case."""

import torch

from cases.softmax_ce import config


def reference_forward(inputs, params):
    """Mean softmax cross-entropy loss using basic tensor ops.

    Inputs:
      logits: [B, C] fp32
      labels: [B] int64
    Output:
      scalar fp32 loss
    """
    logits = inputs["logits"]
    labels = inputs["labels"]

    row_max = logits.max(dim=1, keepdim=True).values
    shifted = logits - row_max
    logsumexp = row_max + torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
    logp = logits - logsumexp
    loss_per_row = -logp.gather(1, labels.view(-1, 1)).squeeze(1)
    return loss_per_row.mean()


def make_inputs(seed, dtype, device, requires_grad=False):
    """Create deterministic logits and integer labels.

    Only logits participates in gradients. labels stays int64.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(config.B, config.C, dtype=dtype, device=device, generator=g)
    labels = torch.randint(0, config.C, (config.B,), dtype=torch.long, device=device, generator=g)

    if requires_grad:
        logits.requires_grad_(True)

    return {"logits": logits, "labels": labels}
