import numpy as np

class EvolveOutcome(object):
    ...

class FFSLevel(object):
    @property
    def configs(self) -> list[np.ndarray]:
        """List of configurations at this level, as arrays (not full states)."""
        ...
    @property
    def previous_indices(self) -> list[int]:
        """For each configuration, the index of the configuration in the previous
        level that resulted in it."""
        ...

class Simulation(object):
    def evolve(
        self,
        state_index: int | None = None,
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min: float | None = None,
        size_max: float | None = None,
        for_wall_time: float | None = None,
        require_strong_bound: bool = True
    ) -> EvolveOutcome: 
        """Evolve a particular state, with index `state_index`,
        subject to some bounds.  Runs state 0 by default.

        By default, this requires a strong bound (the simulation
        will eventually end, eg, not a size or other potentially
        unreachable bound).
        
        Releases the GIL during the simulation."""
        ...
    
    def evolve_all(
        self,
        state_index: int | None = None,
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min: float | None = None,
        size_max: float | None = None,
        for_wall_time: float | None = None,
        require_strong_bound: bool = True
    ) -> list[EvolveOutcome]:
        """Evolve *all* states, stopping each as they reach the
        boundary conditions.  Runs multithreaded using available
        cores.  Runs state 0 by default.
        """
        ...

    def canvas_copy(
        self,
        state_index: int | None
    ) -> np.ndarray:
        "Returns a copy of the state canvas."

    def canvas_view(
        self,
        state_index: int | None
    ) -> np.ndarray:
        """Returns a direct view of the state canvas.  Note that this
        can potentially be unsafe, if the state is later erased."""

    def state_ntiles(
        self,
        state_index: int | None
    ) -> int:
        """Returns the number of tiles in the state."""

    def state_time(
        self,
        state_index: int | None
    ) -> float:
        """Returns the amount of time simulated (in seconds) for the state."""

    def state_events(
        self,
        state_index: int | None
    ) -> int:
        """Returns the number of events simulated for the state."""

    def add_state(
        self
    ) -> int:
        """Add a new state, returning its index."""

    def add_n_states(
        self,
        n: int
    ) -> list[int]:
        """Add `n` new states, returning their indices."""
    
    tile_concs: list[float]
    tile_stoics: list[float]

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
    def surfaces(self) -> list[FFSLevel]: ...
    @property
    def previous_indices(self) -> list[list[int]]: ...

class TileSet(object):
    def from_json(self, json: str) -> TileSet: ...
    def from_dict(self, d: dict) -> TileSet: ...
    def from_file(self, path: str) -> TileSet: ...
    def to_simulation(self) -> Simulation: ...
    def run_window(self) -> Simulation: ...
    def run_ffs(
        self,
        varpermean2: float = 1e-4,
        min_configs: int = 1_000,
        max_size: int = 200,
        cutoff_probability: float = 0.99,
        cutoff_surfaces: int = 4,
        min_cutoff_size: int = 30,
        surface_size_step: int = 1,
        surface_init_size: int = 3,
        max_init_events: int = 10_000,
        max_subseq_events: int = 1_000_000,
        max_init_time: float | None = None,
        max_subseq_time: float | None = None,
        keep_surface_configs: bool = False,
    ) -> FFSResult: ...

class Tile(object):
    ...

class FFSRunConfig(object):
    ...