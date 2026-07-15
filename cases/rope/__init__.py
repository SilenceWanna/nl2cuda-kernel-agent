"""Expose the RoPE case to the algorithm-independent framework."""

import os

from framework.case import Case
from cases.rope import config
from cases.rope.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as file:
        return file.read()


CASE = Case(
    name="rope",
    description=_load_description(),
    params={"base": config.BASE},
    grad_inputs=["X"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
