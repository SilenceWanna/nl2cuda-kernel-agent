from pathlib import Path

from framework.case import Case

from .config import DESCRIPTION, NAME
from .op import candidate, forward_only
from .reference import make_inputs, reference_forward


def _load_description():
    path = Path(__file__).with_name("README.md")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DESCRIPTION.strip()


def _build_case():
    description = _load_description()

    values = {
        "name": NAME,
        "description": description,
        "make_inputs": make_inputs,
        "reference_forward": reference_forward,
        "reference": reference_forward,
        "candidate": candidate,
        "forward_only": forward_only,
    }

    try:
        import dataclasses

        if dataclasses.is_dataclass(Case):
            kwargs = {}
            for field in dataclasses.fields(Case):
                if field.name in values:
                    kwargs[field.name] = values[field.name]
            return Case(**kwargs)
    except Exception:
        pass

    try:
        return Case(
            name=NAME,
            description=description,
            make_inputs=make_inputs,
            reference_forward=reference_forward,
        )
    except TypeError:
        return Case(NAME, description, make_inputs, reference_forward)


CASE = _build_case()

__all__ = [
    "CASE",
    "candidate",
    "forward_only",
    "make_inputs",
    "reference_forward",
]
