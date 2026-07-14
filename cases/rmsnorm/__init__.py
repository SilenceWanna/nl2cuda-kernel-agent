import os

from framework.case import Case
from cases.rmsnorm.reference import make_inputs, reference_forward


def _load_description():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(p, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="rmsnorm",
    description=_load_description(),
    params={"eps": 1e-5},
    grad_inputs=["X", "gamma"],
    dtype="float32",
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
