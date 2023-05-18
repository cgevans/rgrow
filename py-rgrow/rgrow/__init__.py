__all__ = [
    "Tile",
    "TileSet",
    "Simulation",
    "EvolveOutcome",
    "FFSLevel",
    "FFSResult",
    "FFSRunConfig",
]

import numpy as np
from rgrow.rgrow import (
    Tile,
    TileSet,
    EvolveOutcome,
    FFSLevel,
    FFSResult,
    FFSRunConfig,
    System,
    State,
)

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt
    import matplotlib.colors


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
    ):
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
            for_events,
            total_events,
            for_time,
            total_time,
            size_min,
            size_max,
            for_wall_time,
            require_strong_bound,
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
    ):
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
        return self.system.evolve_states(
            self.states,
            for_events,
            total_events,
            for_time,
            total_time,
            size_min,
            size_max,
            for_wall_time,
            require_strong_bound,
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
    ):
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
        return self.system.evolve_states(
            [self.states[i] for i in state_indices],
            for_events,
            total_events,
            for_time,
            total_time,
            size_min,
            size_max,
            for_wall_time,
            require_strong_bound,
        )

    def plot_state(
        self, state_index: int = 0, ax: "int | plt.Axes" = None
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

    def canvas_view(self, state_index: int = 0) -> "np.ndarray":
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

    def canvas_copy(self, state_index: int = 0) -> "np.ndarray":
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
        return self.states[state_index].events

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
        return [s.events for s in self.states]

    @property
    def states_time(self) -> List[float]:
        """Returns the amount of time simulated (in seconds) for each state."""
        return [s.time for s in self.states]

    @property
    def states_ntiles(self) -> List[int]:
        """Returns the number of tiles in each state."""
        return [s.ntiles for s in self.states]

    @property
    def tile_concs(self) -> List[float]:
        """Returns the concentration of each tile in the system."""
        return self.system.tile_concs()

    @property
    def tile_names(self) -> List[str]:
        """Returns the names of each tile in the system."""
        return self.system.tile_names()

    @property
    def tile_colors(self) -> List[tuple[int, int, int, int]]:
        """Returns the colors of each tile in the system."""
        return self.system.tile_colors()

    @property
    def tile_stoics(self) -> List[int]:
        """Returns the stoichiometry of each tile in the system."""
        return self.system.tile_stoics()

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
        return self.system.mismatch_array(self.states[state_index])


def _tileset_to_simulation(ts: TileSet) -> Simulation:
    return Simulation(ts)


TileSet.to_simulation = _tileset_to_simulation  # type: ignore
