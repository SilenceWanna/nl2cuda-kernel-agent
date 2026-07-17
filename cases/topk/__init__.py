"""Expose the fixed-k row-wise Top-K case."""

import os

from framework.case import Case
from cases.topk import config
from cases.topk.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as file:
        return file.read()


CASE = Case(
    name="topk",
    description=_load_description(),
    params={"k": config.K},
    grad_inputs=["X"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
