import os

from framework.case import Case
from cases.rbf import config
from cases.rbf.reference import make_inputs, reference_forward


def _load_description():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(p, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="rbf",
    description=_load_description(),
    params={"gamma": config.GAMMA},
    grad_inputs=["X", "Y"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
