from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class SDCStrand:
    concentration: float
    left_glue: str | None = None
    btm_glue: str | None = None
    right_glue: str | None = None
    name: str | None = None
    color: str | None = None


@dataclass
class SDCParams:
    k_f: float
    k_n: float
    k_c: float
    temperature: float
    glue_dg_s: (
        Mapping[str | tuple[str, str], tuple[float, float] | str]
        | Mapping[str, tuple[float, float] | str]
        | Mapping[tuple[str, str], tuple[float, float] | str]
    )
    scaffold: list[str | None] | list[list[str | None]]
    strands: list[SDCStrand]
