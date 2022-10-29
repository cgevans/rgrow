import numpy as np

class FFSConfig(object): ...

class FFSLevel(object):
    @property
    def configs(self) -> list[np.ndarray]: ...

class Simulation(object): ...

class FFSResult(object):
    @property
    def nucleation_rate(self) -> float: ...

    @property
    def forward_vec(self) -> np.ndarray: ...

    @property
    def dimerization_rate(self) -> float: ...

    @property
    def surfaces(self) -> list[FFSLevel]: ...

class TileSet(object):
    def from_json(self, json: str) -> TileSet: ...
    def from_dict(self, d: dict) -> TileSet: ...
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
