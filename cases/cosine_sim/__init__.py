"""cosine_sim case：暴露 CASE 实例。"""

import os

from framework.case import Case
from cases.cosine_sim import config
from cases.cosine_sim.reference import reference_forward, make_inputs


def _load_description():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(p, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="cosine_sim",
    description=_load_description(),
    params={"eps": 1e-8},
    grad_inputs=["A", "B"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
