import torch

from cases.softmax_ce import config


def reference_forward(inputs, params):
    logits = inputs["logits"]
    labels = inputs["labels"]

    row_max = logits.max(dim=1, keepdim=True).values
    shifted = logits - row_max
    exp_shifted = torch.exp(shifted)
    logsumexp = row_max.squeeze(1) + torch.log(exp_shifted.sum(dim=1))

    rows = torch.arange(logits.shape[0], device=logits.device)
    nll = logsumexp - logits[rows, labels]
    return nll.mean()


def _resolve_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def make_inputs(seed, dtype, device, requires_grad=False):
    torch_dtype = _resolve_dtype(dtype)
    g = torch.Generator(device=device).manual_seed(seed)

    logits = torch.randn(
        config.B,
        config.C,
        dtype=torch_dtype,
        device=device,
        generator=g,
    )
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

    return {
        "logits": logits,
        "labels": labels,
    }
