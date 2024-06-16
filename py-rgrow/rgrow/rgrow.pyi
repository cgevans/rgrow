from enum import Enum
from typing import TYPE_CHECKING, Any, Sequence, overload
import numpy as np
import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

class TileShape(Enum):
    Single = ...
    Vertical = ...
    Horizontal = ...

class EvolveOutcome(object): ...

class NeededUpdate(object): ...

class State(object):
    @property
    def canvas_view(self) -> np.ndarray: ...
    def canvas_copy(self) -> np.ndarray: ...
    @property
    def n_tiles(self) -> int: ...
    @property
    def ntiles(self) -> int: ...
    @property
    def time(self) -> float: ...
    @property
    def total_events(self) -> int: ...
    def print_debug(self) -> None: 
        "Print rust Debug string for the state object."
        ...

class System(object):
    @overload
    def evolve(
        self,
        state: State,
        *,
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min=float | None,
        size_max=float | None,
        for_wall_time: float | None = None,
        require_strong_bound: bool = True,
        show_window: bool = False,
        parallel: bool = True,
    ) -> EvolveOutcome: ...
    @overload
    def evolve(
        self,
        state: Sequence[State],
        *,
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min=float | None,
        size_max=float | None,
        for_wall_time: float | None = None,
        require_strong_bound: bool = True,
        show_window: bool = False,
        parallel: bool = True,
    ) -> list[EvolveOutcome]: ...
    def calc_mismatches(self, state: State | FFSStateRef) -> int: ...
    def calc_mismatch_locations(self, state: State | FFSStateRef) -> np.ndarray: ...
    def name_canvas(self, state: State | FFSStateRef) -> np.ndarray: ...
    def color_canvas(self, state: State | np.ndarray | FFSStateRef) -> np.ndarray: ...
    @property
    def tile_names(self) -> list[str]: ...
    @property
    def tile_colors(self) -> np.ndarray: ...
    def update_all(self, state: State, needed: NeededUpdate | None = None) -> None: ...
    def plot_canvas(
        self,
        state: State,
        ax=None,
        annotate_tiles=False,
        annotate_mismatches=False,
        crop=False,
    ) -> "Axes": ...
    def get_param(self, name: str) -> Any: ...
    def set_param(self, name: str, value: Any): ...
    def print_debug(self) -> None: 
        "Print rust Debug string for the system object."
        ...

class FissionHandling(object): ...
class CanvasType(object): ...
class ChunkSize(object): ...
class ChunkHandling(object): ...
class Model(object): ...

class FFSStateRef(object):
    @property
    def canvas_view(self) -> np.ndarray: ...
    def canvas_copy(self) -> np.ndarray: ...
    @property
    def n_tiles(self) -> int: ...
    @property
    def total_events(self) -> int: ...
    @property
    def time(self) -> float: ...
    def tracking_copy(self) -> np.ndarray: ...
    def clone_state(self) -> State: ...
    def rate_at_point(self, point: tuple[int, int]) -> float: ...
    

class FFSLevelRef(object):
    @property
    def configs(self) -> list[np.ndarray]:
        """List of configurations at this level, as arrays (not full states)."""
        ...
    @property
    def states(self) -> list[FFSStateRef]:
        """List of states at this level."""
        ...
    @property
    def previous_indices(self) -> list[int]:
        """For each configuration, the index of the configuration in the previous
        level that resulted in it."""
        ...

class FFSResult(object):
    @property
    def nucleation_rate(self) -> float:
        """
        The calculated nucleation rate, in M/s.
        """
        ...
    @property
    def forward_vec(self) -> np.ndarray: ...
    @property
    def dimerization_rate(self) -> float: ...
    @property
    def surfaces(self) -> list[FFSLevelRef]: ...
    @property
    def previous_indices(self) -> list[list[int]]: ...
    def configs_dataframe(self) -> pl.DataFrame: ...
    def surfaces_dataframe(self) -> pl.DataFrame: ...

class Tile(object):
    def __init__(
        self,
        bonds: list[str | int],
        name: str | None = None,
        stoic: float | None = None,
        color: str | None = None,
    ) -> None: ...

class FFSRunConfig(object):
    @property
    def constant_variance(self) -> bool: ...
    @property
    def var_per_mean2(self) -> float: ...
    @property
    def min_configs(self) -> int: ...
    @property
    def max_configs(self) -> int: ...
    @property
    def early_cutoff(self) -> bool: ...
    @property
    def cutoff_probability(self) -> float: ...
    @property
    def cutoff_number(self) -> int: ...
    @property
    def min_cutoff_size(self) -> int: ...
    @property
    def init_bound(self) -> EvolveBounds: ...
    @property
    def subseq_bound(self) -> EvolveBounds: ...
    @property
    def start_size(self) -> int: ...
    @property
    def size_step(self) -> int: ...
    @property
    def keep_configs(self) -> bool: ...
    @property
    def min_nuc_rate(self) -> float | None: ...
    @property
    def canvas_size(self) -> tuple[int, int]: ...
    @property
    def target_size(self) -> int: ...

class TileSet(object):
    def __init__(self, **kwargs) -> None: ...
    def create_system(self, **kwargs) -> System: ...
    def create_system_and_state(self, **kwargs) -> tuple[System, State]: ...
    def create_state(self, **kwargs) -> State: ...
    def create_state_empty(self, **kwargs) -> State: ...
    def run_window(self, **kwargs) -> tuple[System, State]: ...
    def run_ffs(
        self,
        constant_variance: bool,
        var_per_mean2: float,
        min_configs: int,
        max_configs: int,
        early_cutoff: bool,
        cutoff_probability: float,
        cutoff_number: int,
        min_cutoff_size: int,
        init_bound: EvolveBounds,
        subseq_bound: EvolveBounds,
        start_size: int,
        size_step: int,
        keep_configs: bool,
        min_nuc_rate: float | None,
        canvas_size: tuple[int, int],
        target_size: int,
        config: FFSRunConfig | None,
        **kwargs,
    ) -> FFSResult: ...

class EvolveBounds(object):
    def __init__(
        self,
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min: float | None = None,
        size_max: float | None = None,
        for_wall_time: float | None = None,
        require_strong_bound: bool = True,
    ) -> None: ...
