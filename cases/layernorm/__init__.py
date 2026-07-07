"""LayerNorm case：暴露 CASE 实例供 framework 使用。"""

import os

from framework.case import Case
from cases.layernorm import config
from cases.layernorm.reference import reference_forward, make_inputs


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="layernorm",
    description=_load_description(),
    params={"eps": config.EPS},
    grad_inputs=["X", "gamma", "beta"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
