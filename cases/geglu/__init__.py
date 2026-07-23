"""GeGLU case exposed to the shared verification and benchmark framework."""

import os

from framework.case import Case
from cases.geglu import config
from cases.geglu.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


CASE = Case(
    name="geglu",
    description=_load_description(),
    params={},
    grad_inputs=["X"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
