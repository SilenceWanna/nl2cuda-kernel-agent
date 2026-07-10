"""Softmax cross-entropy case."""

import os

from framework.case import Case
from cases.softmax_ce.reference import make_inputs, reference_forward


def _load_description():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(p, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="softmax_ce",
    description=_load_description(),
    params={},
    grad_inputs=["logits"],
    dtype="float32",
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
