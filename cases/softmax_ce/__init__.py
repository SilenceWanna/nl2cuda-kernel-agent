"""Softmax cross-entropy case: expose CASE for the framework."""

import os

from framework.case import Case
from cases.softmax_ce import config
from cases.softmax_ce.reference import reference_forward, make_inputs


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="softmax_ce",
    description=_load_description(),
    params={},
    grad_inputs=["logits"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
