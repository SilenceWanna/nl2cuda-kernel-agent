import os

from framework.case import Case
from cases.scan.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as file:
        return file.read()


CASE = Case(
    name="scan",
    description=_load_description(),
    params={},
    grad_inputs=["X"],
    dtype="float32",
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)

