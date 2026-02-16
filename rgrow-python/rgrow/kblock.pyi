from dataclasses import dataclass
from typing import Any, Sequence
from numpy.typing import NDArray
import numpy as np
from matplotlib.axes import Axes
from rgrow import State, FFSStateRef, EvolveOutcome, FFSRunConfig, FFSRunResult


@dataclass
class KBlockTile:
    name: str
    concentration: float
    glues: Sequence[str]
    color: Sequence[int] | str = ...

@dataclass
class KBlockParams:
    tiles: list[KBlockTile]
    blocker_conc: dict[str | int, float]
    seed: dict[tuple[int, int], int | str]
    binding_strength: dict[str, str | float]
    ds_lat: float = ...
    kf: float = ...
    temp: float = ...
    no_partially_blocked_attachments: bool = ...
    blocker_energy_adj: float = ...

class KBlock:
    def __init__(self, params: KBlockParams) -> None: ...
    
    @property
    def tile_names(self) -> list[str]: ...
    @property
    def tile_colors(self) -> NDArray[np.uint8]: ...
    @property
    def bond_names(self) -> list[str]: ...
    @property
    def seed(self) -> dict[tuple[int, int], int]: ...
    @seed.setter
    def seed(self, value: dict[tuple[int, int], int]) -> None: ...
    @property
    def glue_links(self) -> NDArray[np.float64]: ...
    @glue_links.setter
    def glue_links(self, value: NDArray[np.float64]) -> None: ...
    @property
    def cover_concentrations(self) -> list[float]: ...
    @cover_concentrations.setter
    def cover_concentrations(self, value: list[float]) -> None: ...

    def calc_mismatches(self, state: State | FFSStateRef) -> int: ...
    def update_all(self, state: State) -> None: ...
    def evolve(
        self,
        state: State,
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min: int | None = None,
        size_max: int | None = None,
        for_wall_time: float | None = None,
        require_strong_bound: bool = True,
        show_window: bool = False,
        start_window_paused: bool = True,
        parallel: bool = True,
        initial_timescale: float | None = None,
        initial_max_events_per_sec: int | None = None,
    ) -> "EvolveOutcome": ...
    
    def plot_canvas(
        self,
        state: State | np.ndarray | FFSStateRef,
        ax: Axes | None = None,
        annotate_tiles: bool = False,
        annotate_mismatches: bool = False,
        crop: bool = False,
    ) -> Axes: ...
    
    def run_ffs(self, config: FFSRunConfig = ..., **kwargs: Any) -> FFSRunResult: ...

__all__ = ["KBlock", "KBlockParams", "KBlockTile"]

