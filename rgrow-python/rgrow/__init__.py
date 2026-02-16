from __future__ import annotations

__all__ = [
    "Tile",
    "TileSet",
    "Simulation",
    "EvolveOutcome",
    "FFSLevelRef",
    "EvolveBounds",
    "FFSRunResult",
    "FFSRunConfig",
]

import numpy as np
from . import rgrow as rgr
from .rgrow import (
    ATAM,
    KTAM,
    OldKTAM,
    TileSet as _TileSet,
    EvolveOutcome,
    # FFSLevel,
    FFSRunResult,
    FFSLevelRef,
    FFSRunConfig,
    State,
    EvolveBounds,
    FFSStateRef,
)
import attrs
import attr

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
)

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt
    import matplotlib.colors
    import matplotlib.axes
    from numpy.typing import NDArray
    from .kblock import KBlock
    from .rgrow import SDC
System: TypeAlias = "ATAM | KTAM | OldKTAM | KBlock | SDC"
SYSTEMS = (ATAM, KTAM, OldKTAM)


def _system_plot_canvas(
    sys: System,
    state: State | np.ndarray | FFSStateRef,
    ax: matplotlib.axes.Axes | None = None,
    annotate_tiles: bool = False,
    annotate_mismatches: bool = False,
    crop: bool = False,
) -> "plt.Axes":
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    if isinstance(state, (State, FFSStateRef)):
        cv = state.canvas_view
    else:
        cv = state

    rows, cols = cv.shape

    if crop:
        nz = np.nonzero(cv)
        i_min, i_max = nz[0].min(), nz[0].max()
        j_min, j_max = nz[1].min(), nz[1].max()
        rows = i_max - i_min + 1
        cols = j_max - j_min + 1
    else:
        i_min, j_min = 0, 0
        i_max, j_max = rows - 1, cols - 1

    i_grid = i_min + np.tile([0.9, 0.1], rows + 1).cumsum() - 1.45
    j_grid = j_min + np.tile([0.9, 0.1], cols + 1).cumsum() - 1.45

    colors = sys.color_canvas(state)
    if crop:
        colors = colors[i_min : i_max + 1, j_min : j_max + 1, :]

    fullcolors = np.zeros((2 * rows + 1, 2 * cols + 1, 4), dtype=np.uint8)
    fullcolors[1:-1:2, 1:-1:2, :] = colors
    # mask all transparent values
    mask = fullcolors[..., 3] == 0
    mask[0::2, 0::2] = True  # FIXME: remove if adding bond colors
    mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
    fullcolors = np.ma.array(fullcolors, mask=mask, copy=False)

    ax.pcolor(j_grid, i_grid, fullcolors, zorder=2)

    # reverse y
    ax.set_ylim(i_max + 0.5, i_min - 0.7)
    ax.set_xlim(j_min - 0.7, j_max + 0.5)
    ax.set_aspect("equal")
    # Add x ticks and labels at the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Remove bottom and right spines
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Face ticks in
    # ax.xaxis.set_tick_params(direction='in')
    # ax.yaxis.set_tick_params(direction='in')

    if annotate_tiles:
        # If we're annotating, we assume we have space for more ticks!
        ax.set_xticks(np.arange(j_min, j_max + 1, 1))
        ax.set_yticks(np.arange(i_min, i_max + 1, 1))

        # Put light gray grid lines in the background
        ax.grid(True, color="lightgray", zorder=1)

        names = sys.name_canvas(state)
        # colors is already cropped if crop=True, so compute luminance on cropped colors
        tile_colors = colors / 255.0
        lumcolors = np.where(
            tile_colors <= 0.03928,
            tile_colors / 12.92,
            ((tile_colors + 0.055) / 1.055) ** 2.4,
        )
        lum = (
            0.2126 * lumcolors[:, :, 0]
            + 0.7152 * lumcolors[:, :, 1]
            + 0.0722 * lumcolors[:, :, 2]
        )
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                if cv[i, j] == 0:
                    continue
                n = names[i, j]
                # Use relative indices for cropped lum array
                if lum[i - i_min, j - j_min] > 0.2:
                    ax.text(j, i, n, ha="center", va="center", color="black")
                else:
                    ax.text(j, i, n, ha="center", va="center", color="white")

    if annotate_mismatches:
        if isinstance(state, np.ndarray):
            raise ValueError("Cannot currently annotate mismatches on a numpy array.")
        mml = sys.calc_mismatch_locations(state)
        for i, j in zip(*mml.nonzero()):
            d = mml[i, j]
            # We check only 0b1 (west) and 0b10 (south), as 0b100 (east) and 0b1000 (north)
            # will be covered by the tile on the other side of the mismatch.
            if int(d) & 1:  # W
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.75, i - 0.25),
                        0.5,
                        0.5,
                        fill=True,
                        color="red",
                        zorder=3,
                        linewidth=0,
                    )
                )
            if int(d) & 2:  # S
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.25, i + 0.25),
                        0.5,
                        0.5,
                        fill=True,
                        color="red",
                        zorder=3,
                        linewidth=0,
                    )
                )

    return ax


for sys in SYSTEMS:
    sys.plot_canvas = _system_plot_canvas  # type: ignore


@attr.define(auto_attribs=True)
class CoverStrand:
    name: Optional[str]
    glue: int | str
    dir: str
    stoic: float


@attr.define(auto_attribs=True)
class Tile:
    edges: List[int | str]
    name: Optional[str] = None
    stoic: Optional[float] = 1.0
    color: Optional[str] = None
    shape: Optional[str | rgr.TileShape] = None


@attrs.define(auto_attribs=True)
class Bond:
    name: str
    strength: float

    @staticmethod
    def _conv(a: "Bond | Any", *args: Any, **kwargs: Any) -> "Bond":
        if isinstance(a, Bond):
            return a
        elif isinstance(a, dict):
            return Bond(**a)
        elif isinstance(a, tuple):
            return Bond(*a)
        else:
            return Bond(a, *args, **kwargs)  # type: ignore


def _conv_bond_list(q: Sequence[Any]):
    return [Bond._conv(x) for x in q]


@attrs.define(auto_attribs=True)
class TileSet:
    tiles: List[Tile] = attrs.field(factory=list)
    bonds: List[Bond] = attrs.field(
        factory=list, converter=_conv_bond_list, on_setattr=attrs.setters.convert
    )
    glues: List[tuple[int | str, int | str, float]] = attrs.field(factory=list)
    cover_strands: Optional[List[CoverStrand]] = None
    gse: Optional[float] = None
    gmc: Optional[float] = None
    alpha: Optional[float] = None
    threshold: Optional[float] = None
    seed: Optional[tuple[int, int, int | str] | list[tuple[int, int, int | str]]] = None
    size: Optional[int | tuple[int, int]] = None
    tau: Optional[float] = None
    smax: Optional[int] = None
    update_rate: Optional[int] = None
    kf: Optional[float] = None
    fission: Optional[str] = None
    block: Optional[int] = None
    chunk_handling: Optional[str] = None
    chunk_size: Optional[str] = None
    canvas_type: Optional[str] = None
    tracking: Optional[str] = None
    hdoubletiles: Optional[List[Tuple[str | int, str | int]]] = None
    vdoubletiles: Optional[List[Tuple[str | int, str | int]]] = None
    model: Optional[str] = None

    def _to_rg_tileset(self) -> _TileSet:
        kwargs = {
            k: getattr(self, k)
            for k in [
                "gse",
                "gmc",
                "alpha",
                "threshold",
                "seed",
                "size",
                "tau",
                "smax",
                "update_rate",
                "kf",
                "fission",
                "block",
                "chunk_handling",
                "chunk_size",
                "canvas_type",
                "tracking",
                "cover_strands",
                "hdoubletiles",
                "vdoubletiles",
                "model",
            ]
            if getattr(self, k) is not None
        }

        return _TileSet(tiles=self.tiles, bonds=self.bonds, glues=self.glues, **kwargs)

    def create_system(self) -> "System":
        return self._to_rg_tileset().create_system()

    def create_state_empty(self) -> State:
        return self._to_rg_tileset().create_state_empty()

    def create_system_and_state(self) -> tuple[System, State]:
        return self._to_rg_tileset().create_system_and_state()

    def create_state(self, system: System | None = None) -> State:
        if system is None:
            system = self.create_system()
        return self._to_rg_tileset().create_state(system=system)

    def run_window(self) -> State:
        return self._to_rg_tileset().run_window()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TileSet":
        if "tiles" in d:
            d["tiles"] = [Tile(**x) for x in d["tiles"]]
        if "bonds" in d:
            d["bonds"] = [Bond(**x) for x in d["bonds"]]
        if "options" in d:
            d.update(d.pop("options"))
        if "Gse" in d:
            d["gse"] = d.pop("Gse")
        if "Gmc" in d:
            d["gmc"] = d.pop("Gmc")
        # remove unknown keys, and warn
        for k in list(d.keys()):
            if k not in cls.__dict__:
                print(f"Warning: unknown key {k!r} in tileset")
                del d[k]
        return cls(**d)

    def run_ffs(
        self,
        constant_variance: bool = True,
        var_per_mean2: float = 0.01,
        min_configs: int = 100,
        max_configs: int = 100000,
        early_cutoff: bool = True,
        cutoff_probability: float = 0.99,
        cutoff_number: int = 4,
        min_cutoff_size: int = 30,
        init_bound: EvolveBounds = EvolveBounds(for_time=1e7),
        subseq_bound: EvolveBounds = EvolveBounds(for_time=1e7),
        start_size: int = 3,
        size_step: int = 1,
        keep_configs: bool = False,
        min_nuc_rate: float | None = None,
        canvas_size: tuple[int, int] = (64, 64),
        target_size: int = 100,
        config: FFSRunConfig | None = None,  # FIXME
        **kwargs: Any,
    ) -> FFSRunResult:
        return self._to_rg_tileset().run_ffs(
            constant_variance=constant_variance,
            var_per_mean2=var_per_mean2,
            min_configs=min_configs,
            max_configs=max_configs,
            early_cutoff=early_cutoff,
            cutoff_probability=cutoff_probability,
            cutoff_number=cutoff_number,
            min_cutoff_size=min_cutoff_size,
            init_bound=init_bound,
            subseq_bound=subseq_bound,
            start_size=start_size,
            size_step=size_step,
            keep_configs=keep_configs,
            min_nuc_rate=min_nuc_rate,
            canvas_size=canvas_size,
            target_size=target_size,
            **kwargs,
        )


class Simulation:
    tileset: TileSet
    system: System
    states: "List[State]"

    def __init__(
        self,
        tileset: TileSet,
        system: System | None = None,
        states: "List[State] | None" = None,
    ):
        self.tileset = tileset
        self.system = system or tileset.create_system()
        self.states = states or []

    def ensure_state(self, n: int = 0) -> int:
        """Ensure that the simulation has at least n states."""
        while len(self.states) < n:
            self.states.append(self.tileset.create_state())

        return n

    def check_state(self, n: int = 0) -> int:
        """Check that the simulation has at least n states."""
        if len(self.states) < n:
            raise ValueError(
                f"Simulation has {len(self.states)} states, but {n} were required."
            )

        return n

    def evolve(
        self,
        state_index: int = 0,
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min: int | None = None,
        size_max: int | None = None,
        for_wall_time: float | None = None,
        require_strong_bound: bool = True,
    ) -> EvolveOutcome | List[EvolveOutcome]:
        """Evolve a particular state, with index `state_index`,
        subject to some bounds.  Runs state 0 by default.

        By default, this requires a strong bound (the simulation
        will eventually end, eg, not a size or other potentially
        unreachable bound). Releases the GIL during the simulation.

        Parameters
        ----------
        state_index : int, optional
           The index of the state to evolve.  Defaults to 0, and creates sufficient states
           if they do not already exist.
        for_events : int, optional
           Evolve until this many events have occurred.  Defaults to no limit. (Strong bound)
        total_events : int, optional
           Evolve until this many events have occurred in total.  Defaults to no limit.
           (Strong bound)
        for_time : float, optional
           Evolve until this much (physical) time has passed.  Defaults to no limit. (Strong bound)
        total_time : float, optional
           Evolve until this much (physical) time has passed since the state creation.
           Defaults to no limit. (Strong bound)
        size_min : int, optional
           Evolve until the system has this many, or fewer, tiles. Defaults to no limit.
           (Weak bound)
        size_max : int, optional
           Evolve until the system has this many, or more, tiles. Defaults to no limit. (Weak bound)
        for_wall_time : float, optional
           Evolve until this much (wall) time has passed.  Defaults to no limit. (Strong bound)
        require_strong_bound : bool, optional
           If True (default), a ValueError will be raised unless at least one strong bound has been
           set, ensuring that the simulation will eventually end.  If False, ensure only that some
           weak bound has been set, which may result in an infinite simulation.

        Returns
        -------

        EvolveOutcome
          The stopping condition that caused the simulation to end.
        """
        self.ensure_state(state_index)
        state = self.states[state_index]
        return self.system.evolve(
            state,
            for_events=for_events,
            total_events=total_events,
            for_time=for_time,
            total_time=total_time,
            size_min=size_min,
            size_max=size_max,
            for_wall_time=for_wall_time,
            require_strong_bound=require_strong_bound,
        )

    def evolve_all(
        self,
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min: int | None = None,
        size_max: int | None = None,
        for_wall_time: float | None = None,
        require_strong_bound: bool = True,
    ) -> List[EvolveOutcome]:
        """Evolve *all* states, stopping each as they reach the
        boundary conditions.  Runs multithreaded using available
        cores.

        By default, this requires a strong bound (the simulation
        will eventually end, eg, not a size or other potentially
        unreachable bound). Releases the GIL during the simulation.
        Bounds are applied for each state individually.

        Parameters
        ----------
        for_events : int, optional
           Evolve until this many events have occurred.  Defaults to no limit. (Strong bound)
        total_events : int, optional
           Evolve until this many events have occurred in total.  Defaults to no limit.
            (Strong bound)
        for_time : float, optional
           Evolve until this much (physical) time has passed.  Defaults to no limit. (Strong bound)
        total_time : float, optional
           Evolve until this much (physical) time has passed since the state creation.
           Defaults to no limit. (Strong bound)
        size_min : int, optional
           Evolve until the system has this many, or fewer, tiles. Defaults to no limit.
           (Weak bound)
        size_max : int, optional
           Evolve until the system has this many, or more, tiles. Defaults to no limit. (Weak bound)
        for_wall_time : float, optional
           Evolve until this much (wall) time has passed.  Defaults to no limit. (Strong bound)
        require_strong_bound : bool, optional
           If True (default), a ValueError will be raised unless at least one strong bound has been
           set, ensuring that the simulation will eventually end.  If False, ensure only that some
           weak bound has been set, which may result in an infinite simulation.

        Returns
        -------

        list[EvolveOutcome]
          The stopping condition that caused each simulation to end.
        """
        return self.system.evolve(
            self.states,
            for_events=for_events,
            total_events=total_events,
            for_time=for_time,
            total_time=total_time,
            size_min=size_min,
            size_max=size_max,
            for_wall_time=for_wall_time,
            require_strong_bound=require_strong_bound,
        )

    def evolve_some(
        self,
        state_indices: List[int],
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min: int | None = None,
        size_max: int | None = None,
        for_wall_time: float | None = None,
        require_strong_bound: bool = True,
    ) -> EvolveOutcome | List[EvolveOutcome]:
        """
        Evolve *some* states, stopping each as they reach the
        boundary conditions.  Runs multithreaded using available
        cores.

        By default, this requires a strong bound (the simulation
        will eventually end, eg, not a size or other potentially
        unreachable bound). Releases the GIL during the simulation.
        Bounds are applied for each state individually.

        Parameters
        ----------
        state_indices : list[int]
          The indices of the states to evolve.
        for_events : int, optional
           Evolve until this many events have occurred.  Defaults to no limit. (Strong bound)
        total_events : int, optional
           Evolve until this many events have occurred in total.  Defaults to no limit.
           (Strong bound)
        for_time : float, optional
           Evolve until this much (physical) time has passed.  Defaults to no limit. (Strong bound)
        total_time : float, optional
           Evolve until this much (physical) time has passed since the state creation.
           Defaults to no limit. (Strong bound)
        size_min : int, optional
           Evolve until the system has this many, or fewer, tiles. Defaults to no limit.
           (Weak bound)
        size_max : int, optional
           Evolve until the system has this many, or more, tiles. Defaults to no limit. (Weak bound)
        for_wall_time : float, optional
           Evolve until this much (wall) time has passed.  Defaults to no limit. (Strong bound)
        require_strong_bound : bool, optional
           If True (default), a ValueError will be raised unless at least one strong bound has been
           set, ensuring that the simulation will eventually end.  If False, ensure only that some
           weak bound has been set, which may result in an infinite simulation.

        Returns
        -------

        list[EvolveOutcome]
          The stopping condition that caused each simulation to end.
        """
        return self.system.evolve(
            [self.states[i] for i in state_indices],
            for_events=for_events,
            total_events=total_events,
            for_time=for_time,
            total_time=total_time,
            size_min=size_min,
            size_max=size_max,
            for_wall_time=for_wall_time,
            require_strong_bound=require_strong_bound,
        )

    def plot_state(
        self, state_index: int = 0, ax: "plt.Axes | None" = None
    ) -> "plt.QuadMesh":
        """Plot a state as a pcolormesh.  Returns the pcolormesh object."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        v = self.canvas_view(state_index)
        pc = ax.pcolormesh(
            v,
            cmap=self.tile_cmap(),
            linewidth=0.5,
            edgecolors="#ffffff",
        )
        ax.set_aspect("equal")
        ax.set_ylim(v.shape[0], 0)

        return pc

    def tile_cmap(self) -> "matplotlib.colors.ListedColormap":
        """Returns a matplotlib colormap for tile numbers."""
        from matplotlib.colors import ListedColormap
        import numpy as np

        return ListedColormap(
            np.array(self.tile_colors) / 255,
            name="tile_cmap",
        )

    def canvas_view(self, state_index: int = 0) -> NDArray[np.uint]:
        """Returns the current canvas for state_index (default 0), as a
        *direct* view of the state array.  This array will update as
        the simulation evolves.  It should not be modified, as modifications
        will not result in rate and other necessary updates.

        Using this may cause memory safety problems: it is 'unsafe'-labelled in Rust.
        Unless the state is deleted, the array should remain valid so long as the
        underlying Simulation has not been garbage-collected.

        Parameters
        ----------
        state_index : int, optional
           The index of the state to return.  Defaults to 0.

        Returns
        -------

        numpy.ndarray[int]
           The current canvas for the state.
        """
        self.check_state(state_index)
        return self.states[state_index].canvas_view

    def canvas_copy(self, state_index: int = 0) -> NDArray[np.uint]:
        """Returns a copy of the current canvas for state_index (default 0).
        This array will *not* update as the simulation evolves.

        Parameters
        ----------
        state_index : int, optional
           The index of the state to return.  Defaults to 0.

        Returns
        -------

        numpy.ndarray[int]
           The current canvas for the state.
        """
        self.check_state(state_index)
        return self.states[state_index].canvas_copy()

    def name_canvas(self, state_index: int = 0) -> np.ndarray:
        """Returns the current canvas for state_index (default 0), as an array of tile names.
        'empty' indicates empty locations; numbers are translated to strings.

        Parameters
        ----------
        state_index : int, optional
         The index of the state to return.  Defaults to 0.

        Returns
        -------

        numpy.ndarray[str]
         The current canvas for the state, as an array of tile names.
        """
        self.check_state(state_index)
        return self.system.name_canvas(self.states[state_index])

    def state_ntiles(self, state_index: int = 0) -> int:
        """Returns the number of tiles in the state."""
        self.check_state(state_index)
        return self.states[state_index].ntiles

    def state_time(self, state_index: int = 0) -> float:
        """Returns the amount of time simulated (in seconds) for the state."""
        self.check_state(state_index)
        return self.states[state_index].time

    def state_events(self, state_index: int = 0) -> int:
        """Returns the number of events simulated for a state."""
        self.check_state(state_index)
        return self.states[state_index].total_events

    def add_state(self) -> int:
        """Add a state to the simulation."""
        self.states.append(self.tileset.create_state())
        return len(self.states) - 1

    def add_n_states(self, n: int) -> List[int]:
        """Add n states to the simulation."""
        for _ in range(n):
            self.add_state()
        return list(range(len(self.states) - n, len(self.states)))

    @property
    def states_events(self) -> List[int]:
        """Returns the number of events simulated for each state."""
        return [s.total_events for s in self.states]

    @property
    def states_time(self) -> List[float]:
        """Returns the amount of time simulated (in seconds) for each state."""
        return [s.time for s in self.states]

    @property
    def states_ntiles(self) -> List[int]:
        """Returns the number of tiles in each state."""
        return [s.ntiles for s in self.states]

    # @property
    # def tile_concs(self) -> List[float]:
    #     """Returns the concentration of each tile in the system."""
    #     return self.system.tile_concs()

    @property
    def tile_names(self) -> List[str]:
        """Returns the names of each tile in the system."""
        return self.system.tile_names

    @property
    def tile_colors(self) -> np.ndarray:
        """Returns the colors of each tile in the system."""
        return self.system.tile_colors

    # @property
    # def tile_stoics(self) -> List[int]:
    #     """Returns the stoichiometry of each tile in the system."""
    #     return self.system.tile_stoics()

    def set_system_param(self, name: str, value: Any) -> None:
        """Sets a system parameter to a value."""
        upd = self.system.set_param(name, value)
        # FIXME: make this faster
        for state in self.states:
            self.system.update_all(state, upd)

    def get_system_param(self, name: str) -> Any:
        """Gets a system parameter."""
        return self.system.get_param(name)

    def n_mismatches(self, state_index: int = 0) -> int:
        """Returns the number of mismatches in the state."""
        self.check_state(state_index)
        return self.system.calc_mismatches(self.states[state_index])

    def mismatch_array(self, state_index: int = 0) -> np.ndarray:
        """Returns an array of mismatches in the state."""
        self.check_state(state_index)
        return self.system.calc_mismatch_locations(self.states[state_index])


def _tileset_to_simulation(ts: TileSet) -> Simulation:
    return Simulation(ts)


TileSet.to_simulation = _tileset_to_simulation  # type: ignore
