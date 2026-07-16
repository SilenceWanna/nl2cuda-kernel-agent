"""Welford LayerNorm case definition."""

import os

from framework.case import Case
from cases.welford import config
from cases.welford.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as description_file:
        return description_file.read()


CASE = Case(
    name="welford",
    description=_load_description(),
    params={"eps": config.EPS},
    grad_inputs=["X", "gamma", "beta"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
