"""GroupNorm case exposed through the framework Case protocol."""

import os

from framework.case import Case
from cases.groupnorm import config
from cases.groupnorm.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


CASE = Case(
    name="groupnorm",
    description=_load_description(),
    params={"groups": config.GROUPS, "eps": config.EPS},
    grad_inputs=["X", "gamma", "beta"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
