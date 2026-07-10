import torch

from cases.softmax_ce import config


def reference_forward(inputs, params):
    logits = inputs["logits"]
    labels = inputs["labels"]

    row_max = logits.max(dim=1, keepdim=True).values
    shifted = logits - row_max
    logsumexp = row_max.squeeze(1) + torch.log(torch.exp(shifted).sum(dim=1))
    rows = torch.arange(logits.size(0), device=logits.device)
    correct = logits[rows, labels]
    return (logsumexp - correct).mean()


def make_inputs(seed, dtype, device, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        config.B,
        config.C,
        dtype=dtype,
        device=device,
        generator=g,
    )
    labels = torch.randint(
        0,
        config.C,
        (config.B,),
        dtype=torch.int64,
        device=device,
        generator=g,
    )
    if requires_grad:
        logits.requires_grad_(True)
    return {"logits": logits, "labels": labels}
