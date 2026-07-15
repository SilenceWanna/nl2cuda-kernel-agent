import os

from framework.case import Case
from cases.linear_ssm import config
from cases.linear_ssm.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as file:
        return file.read()


CASE = Case(
    name="linear_ssm",
    description=_load_description(),
    params={"a": config.A, "b_coef": config.B_COEF},
    grad_inputs=["X"],
    dtype="float32",
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
