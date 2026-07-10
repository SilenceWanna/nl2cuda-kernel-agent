"""LayerNorm case."""

import os

from framework.case import Case
from cases.layernorm import config
from cases.layernorm.reference import make_inputs, reference_forward


def _load_description():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(p, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="layernorm",
    description=_load_description(),
    params={"eps": config.EPS},
    grad_inputs=["X", "gamma", "beta"],
    dtype="float32",
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
