"""CSR sparse matrix-dense matrix multiplication case definition."""

import os

from framework.case import Case
from cases.spmv import config
from cases.spmv.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as description_file:
        return description_file.read()


CASE = Case(
    name="spmv",
    description=_load_description(),
    params={},
    grad_inputs=["vals", "X"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
