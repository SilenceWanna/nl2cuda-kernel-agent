import os

from framework.case import Case
from cases.gemm_bias_gelu import config
from cases.gemm_bias_gelu.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="gemm_bias_gelu",
    description=_load_description(),
    params={},
    grad_inputs=["X", "W", "b"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)

