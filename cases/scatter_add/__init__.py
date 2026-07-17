"""Scatter-add case definition."""

import os

from framework.case import Case
from cases.scatter_add import config
from cases.scatter_add.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as description_file:
        return description_file.read()


CASE = Case(
    name="scatter_add",
    description=_load_description(),
    params={"S": config.S},
    grad_inputs=["X"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
