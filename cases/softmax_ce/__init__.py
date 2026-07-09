from pathlib import Path

from framework.case import Case

from .config import DESCRIPTION, NAME
from .op import candidate, forward_only
from .reference import make_inputs, reference_forward


def _load_description():
    path = Path(__file__).with_name("README.md")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DESCRIPTION.strip()


CASE = Case(
    name=NAME,
    description=_load_description(),
    params={},
    grad_inputs=["logits"],
    dtype="float32",
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)

__all__ = [
    "CASE",
    "candidate",
    "forward_only",
    "make_inputs",
    "reference_forward",
]
