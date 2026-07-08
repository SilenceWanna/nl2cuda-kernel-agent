"""Softmax 交叉熵 case：暴露 CASE 实例。"""

import os

from framework.case import Case
from cases.softmax_ce import config
from cases.softmax_ce.reference import reference_forward, make_inputs


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="softmax_ce",
    description=_load_description(),
    params={},                       # 无标量参数
    grad_inputs=["logits"],          # 只对 logits 求梯度；labels 是整型索引
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
