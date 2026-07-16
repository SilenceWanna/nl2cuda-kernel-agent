"""Causal attention case definition."""

import os

from framework.case import Case
from cases.causal_attn import config
from cases.causal_attn.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="causal_attn",
    description=_load_description(),
    params={},
    grad_inputs=["Q", "K", "V"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
