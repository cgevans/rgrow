from dataclasses import dataclass, field
from typing import Sequence

@dataclass
class KCovTile:
    name: str
    concentration: float
    glues: Sequence[Sequence[str]]
    color: Sequence[int] | str = field(default_factory=lambda: [100, 100, 100, 255])


@dataclass
class KCovParams:
    tiles: list[KCovTile]
    cover_conc: dict[str | int, float]
    seed: dict[tuple[int, int], int | str]
    binding_strength: dict[str, str | float]
    alpha: float = -7.1
    kf: float = 1e6
    temp: float = 50.0
