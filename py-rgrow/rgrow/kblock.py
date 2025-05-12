from dataclasses import dataclass, field
from typing import Sequence

@dataclass
class KBlockTile:
    name: str
    concentration: float
    glues: Sequence[Sequence[str]]
    color: Sequence[int] | str = field(default_factory=lambda: [100, 100, 100, 255])


@dataclass
class KBlockParams:
    tiles: list[KBlockTile]
    blocker_conc: dict[str | int, float]
    seed: dict[tuple[int, int], int | str]
    binding_strength: dict[str, str | float]
    ds_lat: float = -14.12 / 1000
    kf: float = 1e6
    temp: float = 50.0
    no_partially_blocked_attachments: bool = True
    blocker_energy_adj: float = 0.0
