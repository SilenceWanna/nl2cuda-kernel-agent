"""Segment-softmax case definition."""

import os

from framework.case import Case
from cases.segment_softmax import config
from cases.segment_softmax.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as description_file:
        return description_file.read()


CASE = Case(
    name="segment_softmax",
    description=_load_description(),
    params={"num_segments": config.NUM_SEGMENTS},
    grad_inputs=["values"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
