import torch

from .config import DEFAULT_BATCH, DEFAULT_CLASSES


def reference_forward(inputs, params):
    logits = inputs["logits"]
    labels = inputs["labels"]

    logits_f = logits.float()
    labels = labels.long()

    log_z = torch.logsumexp(logits_f, dim=1)
    correct = logits_f.gather(1, labels.view(-1, 1)).squeeze(1)
    loss = log_z - correct

    return loss.mean()


def make_inputs(seed, dtype, device, requires_grad=False):
    device = torch.device(device)

    try:
        gen = torch.Generator(device=device)
    except RuntimeError:
        gen = torch.Generator(device=device.type)

    gen.manual_seed(seed)

    batch = DEFAULT_BATCH
    classes = DEFAULT_CLASSES

    logits = torch.randn(
        (batch, classes),
        device=device,
        dtype=dtype,
        generator=gen,
    )

    labels = torch.randint(
        low=0,
        high=classes,
        size=(batch,),
        device=device,
        dtype=torch.long,
        generator=gen,
    )

    if requires_grad:
        logits.requires_grad_(True)

    params = {}
    inputs = {
        "logits": logits,
        "labels": labels,
    }

    return inputs, params
