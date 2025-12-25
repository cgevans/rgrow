# flake8: noqa: PYI021
from typing import Any, List, Sequence
from typing_extensions import Self, TypeAlias, overload
from numpy import ndarray
import numpy as np
import polars as pl
from numpy.typing import NDArray
from matplotlib.axes import Axes

class ATAM:
    @property
    def tile_names(self) -> list[str]: ...
    @property
    def tile_colors(self) -> NDArray[np.uint8]: ...
    @property
    def bond_names(self) -> list[str]: ...
    def calc_dimers(self) -> List[DimerInfo]:
        """
        Calculate information about the dimers the system is able to form.

        Returns
        -------
        List[DimerInfo]
        """

    def calc_mismatch_locations(self, state: State | FFSStateRef) -> NDArray[np.uint64]:
        """
        Calculate the locations of mismatches in the state.

        This returns a copy of the canvas, with the values set to 0 if there is no mismatch
        in the location, and > 0, in a model defined way, if there is at least one mismatch.
        Most models use v = 8*N + 4*E + 2*S + W, where N, E, S, W are the four directions.
        Thus, a tile with mismatches to the E and W would have v = 4+2 = 6.

        Parameters
        ----------
        state : State or FFSStateRef
           The state to calculate mismatches for.

        Returns
        -------
        ndarray
          An array of the same shape as the state's canvas, with the values set as described above.
        """

    def calc_mismatches(self, state: State | FFSStateRef) -> int:
        """
        Calculate the number of mismatches in a state.

        Parameters
        ----------
        state : State or FFSStateRef
          The state to calculate mismatches for.

        Returns
        -------
        int
         The number of mismatches.

        See also
        --------
        calc_mismatch_locations
          Calculate the location and direction of mismatches, not jus the number.
        """

    @overload
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
    ) -> EvolveOutcome: ...
    @overload
    def evolve(
        self,
        state: Sequence[State],
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
    ) -> List[EvolveOutcome]: ...
    @overload
    def evolve(
        self,
        state: State | Sequence[State],
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
    ) -> EvolveOutcome | List[EvolveOutcome]:
        """
        Evolve a state (or states), with some bounds on the simulation.

        If evolving multiple states, the bounds are applied per-state.

        Parameters
        ----------
        state : State or Sequence[State]
          The state or states to evolve.
        for_events : int, optional
          Stop evolving each state after this many events.
        total_events : int, optional
          Stop evelving each state when the state's total number of events (including
          previous events) reaches this.
        for_time : float, optional
          Stop evolving each state after this many seconds of simulated time.
        total_time : float, optional
          Stop evolving each state when the state's total time (including previous steps)
          reaches this.
        size_min : int, optional
          Stop evolving each state when the state's number of tiles is less than or equal to this.
        size_max : int, optional
          Stop evolving each state when the state's number of tiles is greater than or equal to this.
        for_wall_time : float, optional
          Stop evolving each state after this many seconds of wall time.
        require_strong_bound : bool
          Require that the stopping conditions are strong, i.e., they are guaranteed to be eventually
          satisfied under normal conditions.
        show_window : bool
          Show a graphical UI window while evolving (requires ui feature, and a single state).
        start_window_paused : bool
          If show_window is True, start the GUI window in a paused state. Defaults to True.
        parallel : bool
          Use multiple threads.
        initial_timescale : float, optional
          If show_window is True, set the initial timescale (sim_time/real_time) in the GUI. None means unlimited.
        initial_max_events_per_sec : int, optional
          If show_window is True, set the initial max events per second limit in the GUI. None means unlimited.

        Returns
        -------
        EvolveOutcome or List[EvolveOutcome]
         The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
        """

    def get_param(self, param_name: str) -> Any: ...
    def print_debug(self) -> None: ...
    @staticmethod
    def read_json(filename: str) -> Self:
        """
        Read a system from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to read from.
        """

    def run_ffs(self, config: FFSRunConfig = ..., **kwargs: Any) -> FFSRunResult:
        """
        Run FFS.

        Parameters
        ----------
        config : FFSRunConfig
         The configuration for the FFS run.
        **kwargs
          FFSRunConfig parameters as keyword arguments.

        Returns
        -------
        FFSRunResult
         The result of the FFS run.
        """

    def set_param(self, param_name: str, value: Any) -> NeededUpdate:
        """
        Set a system parameter.

        Parameters
        ----------
        param_name : str
            The name of the parameter to set.
        value : Any
            The value to set the parameter to.

        Returns
        -------
        NeededUpdate
            The type of state update needed.  This can be passed to
           `update_state` to update the state.
        """

    def tile_color(self, tile_number: int) -> list[int]:
        """
        Given a tile number, return the color of the tile.

        Parameters
        ----------
        tile_number : int
         The tile number.

        Returns
        -------
        list[int]
          The color of the tile, as a list of 4 integers (RGBA).
        """

    def tile_number_from_name(self, tile_name: str) -> int | None:
        """
        Given a tile name, return the tile number.

        Parameters
        ----------
        tile_name : str
          The name of the tile.

        Returns
        -------
        int | None
         The tile number, or None if not found.
        """

    def update_all(self, state: State, needed: NeededUpdate = ...) -> None: ...
    def update_state(self, state: State, needed: NeededUpdate = ...) -> None:
        """
        Recalculate a state's rates.

        This is usually needed when a parameter of the system has
        been changed.

        Parameters
        ----------
        state : State
          The state to update.
        needed : NeededUpdate, optional
          The type of update needed.  If not provided, all locations
          will be recalculated.
        """

    def setup_state(self, state: State) -> None: ...

    def write_json(self, filename: str) -> None:
        """
        Write the system to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        """

    def color_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.uint8]: ...
    def name_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.str_]: ...

    def calc_committer(
        self,
        state: State,
        cutoff_size: int,
        num_trials: int,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float:
        """
        Calculate the committer function for a state.

        Parameters
        ----------
        state : State
            The state to analyze
        cutoff_size : int
            Size threshold for commitment
        num_trials : int
            Number of trials to run
        max_time : float, optional
            Maximum simulation time per trial
        max_events : int, optional
            Maximum events per trial

        Returns
        -------
        float
            Probability of reaching cutoff_size (between 0.0 and 1.0)
        """

    def calc_committer_adaptive(
        self,
        state: State,
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]:
        """
        Calculate the committer function for a state using adaptive sampling.

        Parameters
        ----------
        state : State
            The state to analyze
        cutoff_size : int
            Size threshold for commitment
        conf_interval_margin : float
            Confidence interval margin (e.g., 0.05 for 5%)
        max_time : float, optional
            Maximum simulation time per trial
        max_events : int, optional
            Maximum events per trial

        Returns
        -------
        tuple[float, int]
            Tuple of (probability of reaching cutoff_size, number of trials run)
        """

    def calc_committers_adaptive(
        self,
        states: list[State],
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]:
        """
        Calculate the committer function for multiple states using adaptive sampling.

        Parameters
        ----------
        states : List[State]
            The states to analyze
        cutoff_size : int
            Size threshold for commitment
        conf_interval_margin : float
            Confidence interval margin (e.g., 0.05 for 5%)
        max_time : float, optional
            Maximum simulation time per trial
        max_events : int, optional
            Maximum events per trial

        Returns
        -------
        tuple[NDArray[float64], NDArray[usize]]
            Tuple of (committer probabilities, number of trials for each state)
        """

    def calc_committer_threshold_test(
        self,
        state: State,
        cutoff_size: int,
        threshold: float,
        confidence_level: float,
        max_time: float | None = None,
        max_events: int | None = None,
        max_trials: int | None = None,
        return_on_max_trials: bool = False,
        ci_confidence_level: float | None = None,
    ) -> tuple[bool, float, int, bool]:
        """
        Determine whether the committer probability is above or below a threshold.

        Parameters
        ----------
        state : State
            The state to analyze
        cutoff_size : int
            Size threshold for commitment
        threshold : float
            The probability threshold to compare against
        confidence_level : float
            Confidence level for the threshold test
        max_time : float, optional
            Maximum simulation time per trial
        max_events : int, optional
            Maximum events per trial
        max_trials : int, optional
            Maximum number of trials to run
        return_on_max_trials : bool, optional
            If True, return results even when max_trials is exceeded
        ci_confidence_level : float, optional
            Confidence level for the returned confidence interval (unused)

        Returns
        -------
        tuple[bool, float, int, bool]
            Tuple of (is_above_threshold, probability_estimate, num_trials, exceeded_max_trials)
        """

    def calc_forward_probability(
        self,
        state: State,
        num_trials: int,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float:
        """
        Calculate forward probability for a given state.

        Parameters
        ----------
        state : State
            The initial state to analyze
        num_trials : int
            Number of simulation trials to run
        forward_step : int, optional
            Number of tiles to grow beyond current size (default: 1)
        max_time : float, optional
            Maximum simulation time per trial
        max_events : int, optional
            Maximum number of events per trial

        Returns
        -------
        float
            Probability of reaching forward_step additional tiles
        """

    def calc_forward_probability_adaptive(
        self,
        state: State,
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]:
        """
        Calculate forward probability adaptively for a given state.

        Parameters
        ----------
        state : State
            The initial state to analyze
        conf_interval_margin : float
            Desired confidence interval margin
        forward_step : int, optional
            Number of tiles to grow beyond current size (default: 1)
        max_time : float, optional
            Maximum simulation time per trial
        max_events : int, optional
            Maximum number of events per trial

        Returns
        -------
        tuple[float, int]
            Tuple of (forward probability, number of trials run)
        """

    def calc_forward_probabilities_adaptive(
        self,
        states: list[State],
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]:
        """
        Calculate forward probabilities adaptively for multiple states.

        Parameters
        ----------
        states : list[State]
            List of initial states to analyze
        conf_interval_margin : float
            Desired confidence interval margin
        forward_step : int, optional
            Number of tiles to grow beyond current size (default: 1)
        max_time : float, optional
            Maximum simulation time per trial
        max_events : int, optional
            Maximum number of events per trial

        Returns
        -------
        tuple[NDArray[float64], NDArray[usize]]
            Tuple of (forward probabilities, number of trials for each state)
        """

    def place_tile(self, state: State, point: tuple[int, int], tile: int) -> float:
        """
        Place a tile at a point in the given state.

        Parameters
        ----------
        state : State
            The state to modify.
        point : tuple of int
            The coordinates at which to place the tile (i, j).
        tile : int
            The tile number to place.

        Returns
        -------
        float
            The energy change from placing the tile.
        """

    def find_first_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None:
        """
        Find the first state in a trajectory above the critical threshold.

        Iterates through the trajectory (after filtering redundant events),
        reconstructing the state at each point and testing if the committer
        probability is above the threshold with the specified confidence.

        Parameters
        ----------
        end_state : State
            A state with Movie tracking that contains the trajectory to search.
        config : CriticalStateConfig, optional
            Configuration for the search (uses defaults if not provided)

        Returns
        -------
        CriticalStateResult | None
            The first critical state found, or None if no state is above threshold.
        """

    def find_last_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None:
        """
        Find the last state not above threshold, return the next state.

        Iterates backwards through the trajectory to find the last state that is
        NOT above the critical threshold, then returns the next state (which should
        be above threshold). This is useful for finding the "critical nucleus".

        Parameters
        ----------
        end_state : State
            A state with Movie tracking that contains the trajectory to search.
        config : CriticalStateConfig, optional
            Configuration for the search (uses defaults if not provided)

        Returns
        -------
        CriticalStateResult | None
            The first state above threshold (following the last subcritical state),
            or None if no transition is found.
        """

    def reconstruct_state_from_trajectory(
        self,
        trajectory: pl.DataFrame,
        up_to_index: int,
        config: CriticalStateConfig = ...,
        filter_trajectory: bool = True,
    ) -> State:
        """
        Reconstruct a state from trajectory data up to a given index.

        Parameters
        ----------
        trajectory : pl.DataFrame
            DataFrame with columns: row, col, new_tile
        up_to_index : int
            Index up to which to reconstruct (exclusive)
        config : CriticalStateConfig, optional
            Configuration for state creation
        filter_trajectory : bool, optional
            Whether to filter the trajectory first to remove transient events.
            Default is True.

        Returns
        -------
        State
            The reconstructed state with rates updated.
        """

class SDC:
    def __init__(self, params: Any) -> None: ...
    def mfe_config(self) -> tuple[list[int], float]:
        """
        Calculate the minimum free energy configuration.

        Returns
        -------
        tuple[list[int], float]
            A tuple containing the minimum free energy configuration as a list of tile numbers,
            and the free energy of that configuration.
        """

    @property
    def tile_names(self) -> list[str]: ...
    @property
    def tile_colors(self) -> NDArray[np.uint8]: ...
    @property
    def bond_names(self) -> list[str]: ...

    @property
    def temperature(self) -> float:
        """Temperature in degrees Celsius."""

    @temperature.setter
    def temperature(self, value: float) -> None: ...

    @property
    def scaffold_energy_bonds(self) -> NDArray[np.float64]: ...
    @property
    def strand_energy_bonds(self) -> NDArray[np.float64]: ...
    @property
    def tile_concs(self) -> NDArray[np.float64]: ...
    @tile_concs.setter
    def tile_concs(self, value: list[float]) -> None: ...

    def calc_dimers(self) -> List[DimerInfo]:
        """
        Calculate information about the dimers the system is able to form.

        Returns
        -------
        List[DimerInfo]
        """

    def calc_mismatch_locations(self, state: State | FFSStateRef) -> NDArray[np.uint64]:
        """
        Calculate the locations of mismatches in the state.

        This returns a copy of the canvas, with the values set to 0 if there is no mismatch
        in the location, and > 0, in a model defined way, if there is at least one mismatch.
        Most models use v = 8*N + 4*E + 2*S + W, where N, E, S, W are the four directions.
        Thus, a tile with mismatches to the E and W would have v = 4+2 = 6.

        Parameters
        ----------
        state : State or FFSStateRef
           The state to calculate mismatches for.

        Returns
        -------
        ndarray
          An array of the same shape as the state's canvas, with the values set as described above.
        """

    def calc_mismatches(self, state: State | FFSStateRef) -> int:
        """
        Calculate the number of mismatches in a state.

        Parameters
        ----------
        state : State or FFSStateRef
          The state to calculate mismatches for.

        Returns
        -------
        int
         The number of mismatches.

        See also
        --------
        calc_mismatch_locations
          Calculate the location and direction of mismatches, not jus the number.
        """

    @overload
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
    ) -> EvolveOutcome: ...
    @overload
    def evolve(
        self,
        state: Sequence[State],
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
    ) -> List[EvolveOutcome]: ...
    @overload
    def evolve(
        self,
        state: State | Sequence[State],
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
    ) -> EvolveOutcome | List[EvolveOutcome]:
        """
        Evolve a state (or states), with some bounds on the simulation.

        If evolving multiple states, the bounds are applied per-state.

        Parameters
        ----------
        state : State or Sequence[State]
          The state or states to evolve.
        for_events : int, optional
          Stop evolving each state after this many events.
        total_events : int, optional
          Stop evelving each state when the state's total number of events (including
          previous events) reaches this.
        for_time : float, optional
          Stop evolving each state after this many seconds of simulated time.
        total_time : float, optional
          Stop evolving each state when the state's total time (including previous steps)
          reaches this.
        size_min : int, optional
          Stop evolving each state when the state's number of tiles is less than or equal to this.
        size_max : int, optional
          Stop evolving each state when the state's number of tiles is greater than or equal to this.
        for_wall_time : float, optional
          Stop evolving each state after this many seconds of wall time.
        require_strong_bound : bool
          Require that the stopping conditions are strong, i.e., they are guaranteed to be eventually
          satisfied under normal conditions.
        show_window : bool
          Show a graphical UI window while evolving (requires ui feature, and a single state).
        start_window_paused : bool
          If show_window is True, start the GUI window in a paused state. Defaults to True.
        parallel : bool
          Use multiple threads.
        initial_timescale : float, optional
          If show_window is True, set the initial timescale (sim_time/real_time) in the GUI. None means unlimited.
        initial_max_events_per_sec : int, optional
          If show_window is True, set the initial max events per second limit in the GUI. None means unlimited.

        Returns
        -------
        EvolveOutcome or List[EvolveOutcome]
         The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
        """

    def get_param(self, param_name: str) -> Any: ...
    def print_debug(self) -> None: ...
    @staticmethod
    def read_json(filename: str) -> Self:
        """
        Read a system from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to read from.
        """

    def run_ffs(self, config: FFSRunConfig = ..., **kwargs: Any) -> FFSRunResult:
        """
        Run FFS.

        Parameters
        ----------
        config : FFSRunConfig
         The configuration for the FFS run.
        **kwargs
          FFSRunConfig parameters as keyword arguments.

        Returns
        -------
        FFSRunResult
         The result of the FFS run.
        """

    def set_param(self, param_name: str, value: Any) -> NeededUpdate:
        """
        Set a system parameter.

        Parameters
        ----------
        param_name : str
            The name of the parameter to set.
        value : Any
            The value to set the parameter to.

        Returns
        -------
        NeededUpdate
            The type of state update needed.  This can be passed to
           `update_state` to update the state.
        """

    def tile_color(self, tile_number: int) -> list[int]:
        """
        Given a tile number, return the color of the tile.

        Parameters
        ----------
        tile_number : int
         The tile number.

        Returns
        -------
        list[int]
          The color of the tile, as a list of 4 integers (RGBA).
        """

    def tile_number_from_name(self, tile_name: str) -> int | None:
        """
        Given a tile name, return the tile number.

        Parameters
        ----------
        tile_name : str
          The name of the tile.

        Returns
        -------
        int | None
         The tile number, or None if not found.
        """

    def update_all(self, state: State, needed: NeededUpdate = ...) -> None: ...
    def update_state(self, state: State, needed: NeededUpdate = ...) -> None:
        """
        Recalculate a state's rates.

        This is usually needed when a parameter of the system has
        been changed.

        Parameters
        ----------
        state : State
          The state to update.
        needed : NeededUpdate, optional
          The type of update needed.  If not provided, all locations
          will be recalculated.
        """

    def setup_state(self, state: State) -> None: ...

    def write_json(self, filename: str) -> None:
        """
        Write the system to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        """

    def color_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.uint8]: ...
    def name_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.str_]: ...

    @property
    def entropy_matrix(self) -> NDArray[np.float64]:
        """The ΔS matrix for glue interactions in kcal/mol/K units."""

    @entropy_matrix.setter
    def entropy_matrix(self, value: NDArray[np.float64]) -> None: ...

    @property
    def delta_g_matrix(self) -> NDArray[np.float64]:
        """The ΔG(T=37°C) matrix for glue interactions in kcal/mol units."""

    @delta_g_matrix.setter
    def delta_g_matrix(self, value: NDArray[np.float64]) -> None: ...

    def partition(self) -> float: ...
    def partition_function(self) -> float: ...
    def partition_function_full(self) -> float: ...
    def log_partition_function(self) -> float: ...
    def partial_partition_function(self, constrain_at_loc: list[list[int]]) -> float:
        """
        Calculate the partial partition function with constraints at each location.

        Parameters
        ----------
        constrain_at_loc : list[list[int]]
            A list of lists, where each inner list specifies which tiles are allowed at that
            scaffold position. An empty list means no constraints (all tiles allowed).
            Tile ID 0 corresponds to an empty site.

        Returns
        -------
        float
            The partial partition function value.
        """
    def log_partial_partition_function(self, constrain_at_loc: list[list[int]]) -> float:
        """
        Calculate the log of the partial partition function with constraints at each location.

        Parameters
        ----------
        constrain_at_loc : list[list[int]]
            A list of lists, where each inner list specifies which tiles are allowed at that
            scaffold position. An empty list means no constraints (all tiles allowed).
            Tile ID 0 corresponds to an empty site.

        Returns
        -------
        float
            The log of the partial partition function value.
        """
    def distribution(self) -> list[float]: ...
    def set_tmp_c(self, tmp: float) -> None: ...
    def get_all_probs(self) -> list[tuple[list[int], float, float]]: ...
    def quencher_rates(self) -> str: ...
    def fluorophore_rates(self) -> str: ...
    def probability_of_state(self, state: list[int]) -> float: ...
    def probability_of_constrained_configurations(self, constrain_at_loc: list[list[int]]) -> float:
        """
        Calculate the probability of configurations that satisfy the given constraints.

        Parameters
        ----------
        constrain_at_loc : list[list[int]]
            A list of lists, where each inner list specifies which tiles are allowed at that
            scaffold position. An empty list means no constraints (all tiles allowed).
            Tile ID 0 corresponds to an empty site.

        Returns
        -------
        float
            The probability of configurations satisfying the constraints.
        """
    def state_g(self, state: list[int]) -> float: ...
    def rtval(self) -> float: ...
    def mfe_matrix(self) -> list[list[tuple[int, float, int]]]: ...
    def all_scaffolds_slow(self) -> list[list[int]]: ...

    def calc_committer(
        self,
        state: State,
        cutoff_size: int,
        num_trials: int,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float: ...

    def calc_committer_adaptive(
        self,
        state: State,
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]: ...

    def calc_committers_adaptive(
        self,
        states: list[State],
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]: ...

    def calc_committer_threshold_test(
        self,
        state: State,
        cutoff_size: int,
        threshold: float,
        confidence_level: float,
        max_time: float | None = None,
        max_events: int | None = None,
        max_trials: int | None = None,
        return_on_max_trials: bool = False,
        ci_confidence_level: float | None = None,
    ) -> tuple[bool, float, int, bool]: ...

    def calc_forward_probability(
        self,
        state: State,
        num_trials: int,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float: ...

    def calc_forward_probability_adaptive(
        self,
        state: State,
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]: ...

    def calc_forward_probabilities_adaptive(
        self,
        states: list[State],
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]: ...

    def place_tile(self, state: State, point: tuple[int, int], tile: int) -> float: ...

    def find_first_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None: ...

    def find_last_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None: ...

    def reconstruct_state_from_trajectory(
        self,
        trajectory: pl.DataFrame,
        up_to_index: int,
        config: CriticalStateConfig = ...,
        filter_trajectory: bool = True,
    ) -> State: ...

class EvolveBounds:
    def __init__(
        self,
        for_events: int | None = None,
        total_events: int | None = None,
        for_time: float | None = None,
        total_time: float | None = None,
        size_min: int | None = None,
        size_max: int | None = None,
        for_wall_time: float | None = None,
    ) -> None: ...
    def is_strongly_bounded(self) -> bool: ...
    def is_weakly_bounded(self) -> bool:
        """
        Will the EvolveBounds actually bound anything, or is it just null, such that the simulation will continue
        until a ZeroRate or an error?  Note that this includes weak bounds (size minimum and maximum) that may
        never be reached.
        """

class EvolveOutcome:
    """Outcome of an evolve operation."""
    ReachedEventsMax: EvolveOutcome
    ReachedTimeMax: EvolveOutcome
    ReachedWallTimeMax: EvolveOutcome
    ReachedSizeMin: EvolveOutcome
    ReachedSizeMax: EvolveOutcome
    ReachedZeroRate: EvolveOutcome

class FFSRunConfig:
    """
    Configuration options for Forward Flux Sampling (FFS) simulations.
    
    FFS is a rare event sampling method that calculates nucleation rates by dividing
    the nucleation process into a series of surfaces (levels) based on cluster size,
    then computing the probability of crossing each surface.
    
    Parameters
    ----------
    constant_variance : bool, optional
        Use constant-variance, variable-configurations-per-surface method.
        When True, the number of configurations generated at each surface is determined
        dynamically to achieve a target variance of the forward probability relative to the mean 
        squared (var_per_mean2). When False, exactly max_configs configurations are generated at 
        each surface. Default is True.
    var_per_mean2 : float, optional
        Target variance per mean squared for the constant-variance method.
        Controls the statistical precision when constant_variance is True. Lower values
        require more configurations but provide better statistics. Typical values are
        0.01 (1% variance) to 0.1 (10% variance). Only used when constant_variance is True.
        Default is 0.01.
    min_configs : int, optional
        Minimum number of configurations to generate at each surface level.
        Ensures a minimum sample size even when constant_variance is True and the
        target variance is achieved with fewer configurations. Default is 1000.
    max_configs : int, optional
        Maximum number of configurations to generate at each surface level.
        When constant_variance is False, exactly this many configurations are generated.
        When constant_variance is True, this serves as an upper limit to prevent
        excessive computation when success probabilities are very low. Default is 100000.
    early_cutoff : bool, optional
        Enable early termination when success probabilities become very high.
        When True, FFS will terminate early if the success probability exceeds
        cutoff_probability for cutoff_number consecutive surfaces, provided the
        structure size is at least min_cutoff_size. Default is True.
    cutoff_probability : float, optional
        Success probability threshold for early cutoff.
        If early_cutoff is True and the success probability exceeds this value
        for cutoff_number consecutive surfaces, FFS terminates early. Default is 0.99.
    cutoff_number : int, optional
        Number of consecutive high-probability surfaces required for early cutoff.
        FFS terminates early only after this many consecutive surfaces exceed
        cutoff_probability. Prevents premature termination due to statistical
        fluctuations. Only used when early_cutoff is True. Default is 4.
    min_cutoff_size : int, optional
        Minimum structure size required before early cutoff can occur.
        Prevents early termination when structures are still small, even if success
        probabilities are high. Ensures the simulation reaches a meaningful size
        before terminating. Only used when early_cutoff is True. Default is 30.
    init_bound : EvolveBounds, optional
        Evolution bounds for the initial dimer-to-n-mer surface, to avoid 
        infinite simulations. Default is EvolveBounds(for_time=1e7).
    subseq_bound : EvolveBounds, optional
        Evolution bounds for subsequent surface-to-surface transitions, to avoid
        infinite simulations. Default is EvolveBounds(for_time=1e7).
    start_size : int, optional
        Initial cluster size for the first FFS surface.
        The size (number of tiles) that defines the first surface. Must be >=2.
        Default is 3.
    size_step : int, optional
        Size increment between consecutive FFS surfaces.
        The number of tiles by which the target size increases between consecutive
        surfaces. Default is 1.
    keep_configs : bool, optional
        Whether to retain configuration data for each surface.
        When True, all generated configurations are stored in memory, consuming significant memory
        but allowing state access. When False, only statistics are retained. Default is False.
    min_nuc_rate : float, optional
        Minimum nucleation rate threshold for early termination.
        If specified, FFS terminates early when the calculated nucleation rate
        falls below this threshold. Useful for avoiding excessive computation
        when nucleation rates become negligibly small. Units: M/s. Default is None.
    canvas_size : tuple[int, int], optional
        Canvas dimensions (width, height) for the simulation.
        Defines the size of the 2D lattice on which tile assembly occurs.
        Must be large enough to accommodate the largest expected structures.
        Default is (32, 32).
    canvas_type : Any, optional
        Type of boundary conditions for the simulation canvas.
        Determines how the edges of the canvas are handled:
        - Periodic: opposite edges are connected (torus topology)
        - Square: finite canvas with hard boundaries
        - Tube: periodic in one dimension, finite in the other
        Default is Periodic.
    tracking : str, optional
        Type of additional data tracking during simulation.
        Controls what extra information is recorded during evolution.
        Accepts a string (case-insensitive):
        - "None": no additional tracking (fastest, default)
        - "Order": track attachment order of tiles
        - "LastAttachTime": track when the tile at each location last attached
        - "PrintEvent": print events as they occur (debugging)
        - "Movie": record all events
        Default is "None".
    target_size : int, optional
        Target structure size for FFS termination. Default is 100.
    store_ffs_config : bool, optional
        Whether to store the FFS configuration in the result.
        When True, the complete FFSRunConfig is saved with the results. Default is True.
    store_system : bool, optional
        Whether to store the tile system in the result. Default is False.
    """
    def __init__(
        self,
        constant_variance: bool | None = None,
        var_per_mean2: float | None = None,
        min_configs: int | None = None,
        max_configs: int | None = None,
        early_cutoff: bool | None = None,
        cutoff_probability: float | None = None,
        cutoff_number: int | None = None,
        min_cutoff_size: int | None = None,
        init_bound: EvolveBounds | None = None,
        subseq_bound: EvolveBounds | None = None,
        start_size: int | None = None,
        size_step: int | None = None,
        keep_configs: bool | None = None,
        min_nuc_rate: float | None = None,
        canvas_size: tuple[int, int] | None = None,
        canvas_type: Any | None = None,
        tracking: str | None = None,
        target_size: int | None = None,
        store_ffs_config: bool | None = None,
        store_system: bool | None = None,
    ) -> None: ...

class FFSLevelRef:
    @property
    def configs(self) -> list[NDArray[np.uint]]:
        """list[NDArray[np.uint]]: List of configuration arrays for this surface."""
        
    @property
    def states(self) -> list[FFSStateRef]:
        """list[FFSStateRef]: List of state references for this surface."""
        
    @property
    def previous_indices(self) -> list[int]:
        """list[int]: Previous indices for configurations in this surface."""
    
    def get_state(self, i: int) -> FFSStateRef:
        """
        Get a specific state from this surface.
        
        Parameters
        ----------
        i : int
            Index of the state to retrieve.
            
        Returns
        -------
        FFSStateRef
            The state at the given index.
        """
        
    def has_stored_states(self) -> bool:
        """
        Check if this surface has stored states.
        
        Returns
        -------
        bool
            True if states are stored, False otherwise.
        """

class FFSRunResult:
    @property
    def nucleation_rate(self) -> float:
        """float: Nucleation rate, in M/s. Calculated from the forward probability vector and dimerization rate."""
        
    @property
    def forward_vec(self) -> NDArray[np.float64]:
        """NDArray[np.float64]: Forward probability vector."""
        
    @property
    def dimerization_rate(self) -> float:
        """float: Dimerization rate, in M/s."""
        
    @property
    def surfaces(self) -> list[FFSLevelRef]:
        """list[FFSLevelRef]: List of surfaces."""
        
    @property
    def previous_indices(self) -> list[list[int]]:
        """list[list[int]]: Previous indices for each surface."""
    
    def configs_dataframe(self) -> pl.DataFrame:
        """
        Get the configurations as a Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
        """
        
    def surfaces_dataframe(self) -> pl.DataFrame:
        """
        Get the surfaces as a Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
        """
        
    def surfaces_to_polars(self) -> pl.DataFrame:
        """
        Get the surfaces as a Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
        """
        
    def states_to_polars(self) -> pl.DataFrame:
        """
        Get the states as a Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
        """
    
    def into_resdf(self) -> FFSRunResultDF:
        """Convert to FFSRunResultDF format."""
        
    def write_files(self, prefix: str) -> None:
        """
        Write dataframes and result data to files.
        
        Parameters
        ----------
        prefix : str
           Prefix for the filenames. The files will be named
           `{prefix}.surfaces.parquet`, `{prefix}.configurations.parquet`, and
           `{prefix}.ffs_result.json`.
        """

class FFSRunResultDF:
    @property
    def nucleation_rate(self) -> float:
        """float: Nucleation rate, in M/s. Calculated from the forward probability vector and dimerization rate."""
        
    @property
    def forward_vec(self) -> NDArray[np.float64]:
        """NDArray[np.float64]: Forward probability vector."""
        
    @property
    def dimerization_rate(self) -> float:
        """float: Dimerization rate, in M/s."""
    
    def configs_dataframe(self) -> pl.DataFrame:
        """
        Get the configurations as a Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
        """
        
    @staticmethod
    def read_files(prefix: str) -> FFSRunResultDF:
        """
        Read dataframes and result data from files.

        Parameters
        ----------
        prefix : str
           Prefix for the filenames to read from.

        Returns
        -------
        FFSRunResultDF
        """

    def surfaces_dataframe(self) -> pl.DataFrame:
        """
        Get the surfaces as a Polars DataFrame.

        Returns
        -------
        pl.DataFrame
        """

    def write_files(self, prefix: str) -> None:
        """
        Write dataframes and result data to files.

        Parameters
        ----------
        prefix : str
           Prefix for the filenames.  The files will be named
           `{prefix}.surfaces.parquet`, `{prefix}.configurations.parquet`, and
           `{prefix}.ffs_result.json`.
        """

class CriticalStateConfig:
    """
    Configuration for critical state search algorithms.
    
    Parameters
    ----------
    cutoff_size : int, optional
        Cutoff size for committer calculation (tiles above which growth is considered successful).
        Default is 100.
    threshold : float, optional
        Probability threshold for determining if state is "critical" (above/below this).
        Default is 0.5.
    confidence_level : float, optional
        Confidence level for the threshold test. Default is 0.98.
    max_trials : int, optional
        Maximum number of trials for committer calculation. Default is 100000.
    ci_confidence_level : float, optional
        Confidence level for the confidence interval (if requested). Default is 0.95.
    canvas_size : tuple[int, int], optional
        Canvas size for state reconstruction. Default is (32, 32).
    canvas_type : str, optional
        Canvas type for state reconstruction. Default is "periodic".
    """
    cutoff_size: int
    threshold: float
    confidence_level: float
    max_trials: int
    ci_confidence_level: float
    canvas_size: tuple[int, int]
    canvas_type: str
    
    def __init__(
        self,
        cutoff_size: int | None = None,
        threshold: float | None = None,
        confidence_level: float | None = None,
        max_trials: int | None = None,
        ci_confidence_level: float | None = None,
        canvas_size: tuple[int, int] | None = None,
        canvas_type: str | None = None,
    ) -> None: ...

class CriticalStateResult:
    """
    Result of a critical state search.

    Properties
    ----------
    state : State
        The critical state found.
    energy : float
        Energy at the critical state.
    trajectory_index : int
        Index in the trajectory where the critical state was found.
    is_above_threshold : bool
        Whether the state is above threshold.
    probability : float
        Estimated committer probability.
    num_trials : int
        Number of trials used in the calculation.
    max_trials_exceeded : bool
        Whether max trials was exceeded.
    """
    @property
    def state(self) -> State: ...
    @property
    def energy(self) -> float: ...
    @property
    def trajectory_index(self) -> int: ...
    @property
    def is_above_threshold(self) -> bool: ...
    @property
    def probability(self) -> float: ...
    @property
    def num_trials(self) -> int: ...
    @property
    def max_trials_exceeded(self) -> bool: ...

class FFSStateRef:
    @property
    def time(self) -> float:
        """float: The total time the state has simulated, in seconds."""
        
    @property
    def total_events(self) -> int:
        """int: The total number of events that have occurred in the state."""
        
    @property
    def n_tiles(self) -> int:
        """int: The number of tiles in the state."""
        
    @property
    def canvas_view(self) -> NDArray[np.uint]:
        """NDArray[np.uint]: A direct, mutable view of the state's canvas. This is potentially unsafe."""
        
    @property
    def total_rate(self) -> float:
        """float: The total rate of possible next events for the state."""

    def canvas_copy(self) -> NDArray[np.uint]:
        """
        Create a copy of the state's canvas.
        
        This is safe, but can't be modified and is slower than `canvas_view`.
        
        Returns
        -------
        NDArray[np.uint]
            A copy of the state's canvas.
        """

    def clone_state(self) -> State:
        """
        Return a copy of the state behind the reference as a mutable `State` object.
        
        Returns
        -------
        State
            A mutable copy of the state.
        """
        
    def tracking_copy(self) -> Any:
        """
        Return a copy of the tracker's tracking data.
        
        Returns
        -------
        Any
            The tracking data.
        """
        
    def rate_array(self) -> NDArray[np.float64]:
        """
        Return a cloned copy of an array with the total possible next event rate for each point in the canvas.
        
        This is the deepest level of the quadtree for tree-based states.
        
        Returns
        -------
        NDArray[np.float64]
            Array of rates for each canvas position.
        """

class KTAM:
    @property
    def tile_names(self) -> list[str]: ...
    @property
    def tile_colors(self) -> NDArray[np.uint8]: ...
    @property
    def bond_names(self) -> list[str]: ...

    @property
    def alpha(self) -> float: ...
    @alpha.setter
    def alpha(self, value: float) -> None: ...

    @property
    def g_se(self) -> float: ...
    @g_se.setter
    def g_se(self, value: float) -> None: ...

    @property
    def kf(self) -> float: ...
    @kf.setter
    def kf(self, value: float) -> None: ...

    @property
    def energy_we(self) -> NDArray[np.float64]: ...
    @property
    def energy_ns(self) -> NDArray[np.float64]: ...
    @property
    def tile_concs(self) -> NDArray[np.float64]: ...
    @property
    def tile_edges(self) -> NDArray[np.uint]: ...
    @tile_edges.setter
    def tile_edges(self, value: NDArray[np.uint]) -> None: ...
    @property
    def has_duples(self) -> bool: ...

    def calc_dimers(self) -> List[DimerInfo]:
        """
        Calculate information about the dimers the system is able to form.

        Returns
        -------
        List[DimerInfo]
        """

    def calc_mismatch_locations(self, state: State | FFSStateRef) -> ndarray:
        """
        Calculate the locations of mismatches in the state.

        This returns a copy of the canvas, with the values set to 0 if there is no mismatch
        in the location, and > 0, in a model defined way, if there is at least one mismatch.
        Most models use v = 8*N + 4*E + 2*S + W, where N, E, S, W are the four directions.
        Thus, a tile with mismatches to the E and W would have v = 4+2 = 6.

        Parameters
        ----------
        state : State or FFSStateRef
           The state to calculate mismatches for.

        Returns
        -------
        ndarray
          An array of the same shape as the state's canvas, with the values set as described above.
        """

    def calc_mismatches(self, state: State | FFSStateRef) -> int:
        """
        Calculate the number of mismatches in a state.

        Parameters
        ----------
        state : State or FFSStateRef
          The state to calculate mismatches for.

        Returns
        -------
        int
         The number of mismatches.

        See also
        --------
        calc_mismatch_locations
          Calculate the location and direction of mismatches, not jus the number.
        """

    @overload
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
    ) -> EvolveOutcome: ...
    @overload
    def evolve(
        self,
        state: Sequence[State],
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
    ) -> List[EvolveOutcome]: ...
    @overload
    def evolve(
        self,
        state: State | Sequence[State],
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
    ) -> EvolveOutcome | List[EvolveOutcome]:
        """
        Evolve a state (or states), with some bounds on the simulation.

        If evolving multiple states, the bounds are applied per-state.

        Parameters
        ----------
        state : State or Sequence[State]
          The state or states to evolve.
        for_events : int, optional
          Stop evolving each state after this many events.
        total_events : int, optional
          Stop evelving each state when the state's total number of events (including
          previous events) reaches this.
        for_time : float, optional
          Stop evolving each state after this many seconds of simulated time.
        total_time : float, optional
          Stop evolving each state when the state's total time (including previous steps)
          reaches this.
        size_min : int, optional
          Stop evolving each state when the state's number of tiles is less than or equal to this.
        size_max : int, optional
          Stop evolving each state when the state's number of tiles is greater than or equal to this.
        for_wall_time : float, optional
          Stop evolving each state after this many seconds of wall time.
        require_strong_bound : bool
          Require that the stopping conditions are strong, i.e., they are guaranteed to be eventually
          satisfied under normal conditions.
        show_window : bool
          Show a graphical UI window while evolving (requires ui feature, and a single state).
        start_window_paused : bool
          If show_window is True, start the GUI window in a paused state. Defaults to True.
        parallel : bool
          Use multiple threads.
        initial_timescale : float, optional
          If show_window is True, set the initial timescale (sim_time/real_time) in the GUI. None means unlimited.
        initial_max_events_per_sec : int, optional
          If show_window is True, set the initial max events per second limit in the GUI. None means unlimited.

        Returns
        -------
        EvolveOutcome or List[EvolveOutcome]
         The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
        """

    @staticmethod
    def from_tileset(tileset: Any) -> Self: ...
    def get_param(self, param_name: str) -> Any: ...
    def print_debug(self) -> None: ...
    @staticmethod
    def read_json(filename: str) -> Self:
        """
        Read a system from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to read from.
        """

    def run_ffs(self, config: FFSRunConfig = ..., **kwargs: Any) -> FFSRunResult:
        """
        Run FFS.

        Parameters
        ----------
        config : FFSRunConfig
         The configuration for the FFS run.
        **kwargs
          FFSRunConfig parameters as keyword arguments.

        Returns
        -------
        FFSRunResult
         The result of the FFS run.
        """

    def set_param(self, param_name: str, value: Any) -> NeededUpdate:
        """
        Set a system parameter.

        Parameters
        ----------
        param_name : str
            The name of the parameter to set.
        value : Any
            The value to set the parameter to.

        Returns
        -------
        NeededUpdate
            The type of state update needed.  This can be passed to
           `update_state` to update the state.
        """

    def tile_color(self, tile_number: int) -> list[int]:
        """
        Given a tile number, return the color of the tile.

        Parameters
        ----------
        tile_number : int
         The tile number.

        Returns
        -------
        list[int]
          The color of the tile, as a list of 4 integers (RGBA).
        """

    def tile_number_from_name(self, tile_name: str) -> int | None:
        """
        Given a tile name, return the tile number.

        Parameters
        ----------
        tile_name : str
          The name of the tile.

        Returns
        -------
        int | None
         The tile number, or None if not found.
        """

    def update_all(self, state: State, needed: NeededUpdate = ...) -> None: ...
    def update_state(self, state: State, needed: NeededUpdate = ...) -> None:
        """
        Recalculate a state's rates.

        This is usually needed when a parameter of the system has
        been changed.

        Parameters
        ----------
        state : State
          The state to update.
        needed : NeededUpdate, optional
          The type of update needed.  If not provided, all locations
          will be recalculated.
        """

    def setup_state(self, state: State) -> None: ...

    def write_json(self, filename: str) -> None:
        """
        Write the system to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        """

    def color_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.uint8]: ...
    def name_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.str_]: ...
    def plot_canvas(
        self,
        state: State | np.ndarray | FFSStateRef,
        ax: "Axes" | None = None,
        annotate_tiles: bool = False,
        annotate_mismatches: bool = False,
        crop: bool = False,
    ) -> "Axes": ...
    def total_free_energy_from_point(
        self, state: State, p: tuple[int, int]
    ) -> float:
        """
        Calculate the total free energy contribution from a specific point.  This includes
        the energy from the N and W bonds, and the entropic cost of mixing.
        
        Parameters
        ----------
        state : State
            The state to calculate free energy for.
        p : tuple[int, int]
            The (row, col) coordinates of the point.
            
        Returns
        -------
        float
            The total free energy contribution from the point.
            
        Raises
        ------
        ValueError
            If the point is out of bounds.
        """
    def state_energy(self, state: State) -> float:
        """
        Calculate the total free energy of the entire state.
        
        Parameters
        ----------
        state : State
            The state to calculate energy for.
            
        Returns
        -------
        float
            The total energy of the state.
        """

    def calc_committer(
        self,
        state: State,
        cutoff_size: int,
        num_trials: int,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float: ...

    def calc_committer_adaptive(
        self,
        state: State,
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]: ...

    def calc_committers_adaptive(
        self,
        states: list[State],
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]: ...

    def calc_committer_threshold_test(
        self,
        state: State,
        cutoff_size: int,
        threshold: float,
        confidence_level: float,
        max_time: float | None = None,
        max_events: int | None = None,
        max_trials: int | None = None,
        return_on_max_trials: bool = False,
        ci_confidence_level: float | None = None,
    ) -> tuple[bool, float, int, bool]: ...

    def calc_forward_probability(
        self,
        state: State,
        num_trials: int,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float: ...

    def calc_forward_probability_adaptive(
        self,
        state: State,
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]: ...

    def calc_forward_probabilities_adaptive(
        self,
        states: list[State],
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]: ...

    def place_tile(self, state: State, point: tuple[int, int], tile: int) -> float: ...

    def find_first_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None: ...

    def find_last_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None: ...

    def reconstruct_state_from_trajectory(
        self,
        trajectory: pl.DataFrame,
        up_to_index: int,
        config: CriticalStateConfig = ...,
        filter_trajectory: bool = True,
    ) -> State: ...

class OldKTAM:
    @property
    def tile_names(self) -> list[str]: ...
    @property
    def tile_colors(self) -> NDArray[np.uint8]: ...
    @property
    def bond_names(self) -> list[str]: ...
    def color_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.uint8]: ...
    def name_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.str_]: ...
    def calc_dimers(self) -> List[DimerInfo]:
        """
        Calculate information about the dimers the system is able to form.

        Returns
        -------
        List[DimerInfo]
        """

    def calc_mismatch_locations(self, state: State | FFSStateRef) -> ndarray:
        """
        Calculate the locations of mismatches in the state.

        This returns a copy of the canvas, with the values set to 0 if there is no mismatch
        in the location, and > 0, in a model defined way, if there is at least one mismatch.
        Most models use v = 8*N + 4*E + 2*S + W, where N, E, S, W are the four directions.
        Thus, a tile with mismatches to the E and W would have v = 4+2 = 6.

        Parameters
        ----------
        state : State or FFSStateRef
           The state to calculate mismatches for.

        Returns
        -------
        ndarray
          An array of the same shape as the state's canvas, with the values set as described above.
        """

    def calc_mismatches(self, state: State | FFSStateRef) -> int:
        """
        Calculate the number of mismatches in a state.

        Parameters
        ----------
        state : State or FFSStateRef
          The state to calculate mismatches for.

        Returns
        -------
        int
         The number of mismatches.

        See also
        --------
        calc_mismatch_locations
          Calculate the location and direction of mismatches, not jus the number.
        """

    @overload
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
    ) -> EvolveOutcome: ...
    @overload
    def evolve(
        self,
        state: Sequence[State],
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
    ) -> List[EvolveOutcome]: ...
    @overload
    def evolve(
        self,
        state: State | Sequence[State],
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
    ) -> EvolveOutcome | List[EvolveOutcome]:
        """
        Evolve a state (or states), with some bounds on the simulation.

        If evolving multiple states, the bounds are applied per-state.

        Parameters
        ----------
        state : State or Sequence[State]
          The state or states to evolve.
        for_events : int, optional
          Stop evolving each state after this many events.
        total_events : int, optional
          Stop evelving each state when the state's total number of events (including
          previous events) reaches this.
        for_time : float, optional
          Stop evolving each state after this many seconds of simulated time.
        total_time : float, optional
          Stop evolving each state when the state's total time (including previous steps)
          reaches this.
        size_min : int, optional
          Stop evolving each state when the state's number of tiles is less than or equal to this.
        size_max : int, optional
          Stop evolving each state when the state's number of tiles is greater than or equal to this.
        for_wall_time : float, optional
          Stop evolving each state after this many seconds of wall time.
        require_strong_bound : bool
          Require that the stopping conditions are strong, i.e., they are guaranteed to be eventually
          satisfied under normal conditions.
        show_window : bool
          Show a graphical UI window while evolving (requires ui feature, and a single state).
        start_window_paused : bool
          If show_window is True, start the GUI window in a paused state. Defaults to True.
        parallel : bool
          Use multiple threads.
        initial_timescale : float, optional
          If show_window is True, set the initial timescale (sim_time/real_time) in the GUI. None means unlimited.
        initial_max_events_per_sec : int, optional
          If show_window is True, set the initial max events per second limit in the GUI. None means unlimited.

        Returns
        -------
        EvolveOutcome or List[EvolveOutcome]
         The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
        """

    def get_param(self, param_name: str) -> Any: ...
    def print_debug(self) -> None: ...
    @staticmethod
    def read_json(filename: str) -> Self:
        """
        Read a system from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to read from.
        """

    def run_ffs(self, config: FFSRunConfig = ..., **kwargs: Any) -> FFSRunResult:
        """
        Run FFS.

        Parameters
        ----------
        config : FFSRunConfig
         The configuration for the FFS run.
        **kwargs
          FFSRunConfig parameters as keyword arguments.

        Returns
        -------
        FFSRunResult
         The result of the FFS run.
        """

    def set_param(self, param_name: str, value: Any) -> NeededUpdate:
        """
        Set a system parameter.

        Parameters
        ----------
        param_name : str
            The name of the parameter to set.
        value : Any
            The value to set the parameter to.

        Returns
        -------
        NeededUpdate
            The type of state update needed.  This can be passed to
           `update_state` to update the state.
        """

    def tile_color(self, tile_number: int) -> list[int]:
        """
        Given a tile number, return the color of the tile.

        Parameters
        ----------
        tile_number : int
         The tile number.

        Returns
        -------
        list[int]
          The color of the tile, as a list of 4 integers (RGBA).
        """

    def tile_number_from_name(self, tile_name: str) -> int | None:
        """
        Given a tile name, return the tile number.

        Parameters
        ----------
        tile_name : str
          The name of the tile.

        Returns
        -------
        int | None
         The tile number, or None if not found.
        """

    def update_all(self, state: State, needed: NeededUpdate = ...) -> None: ...
    def update_state(self, state: State, needed: NeededUpdate = ...) -> None:
        """
        Recalculate a state's rates.

        This is usually needed when a parameter of the system has
        been changed.

        Parameters
        ----------
        state : State
          The state to update.
        needed : NeededUpdate, optional
          The type of update needed.  If not provided, all locations
          will be recalculated.
        """

    def setup_state(self, state: State) -> None: ...

    def write_json(self, filename: str) -> None:
        """
        Write the system to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        """

    def calc_committer(
        self,
        state: State,
        cutoff_size: int,
        num_trials: int,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float: ...

    def calc_committer_adaptive(
        self,
        state: State,
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]: ...

    def calc_committers_adaptive(
        self,
        states: list[State],
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]: ...

    def calc_committer_threshold_test(
        self,
        state: State,
        cutoff_size: int,
        threshold: float,
        confidence_level: float,
        max_time: float | None = None,
        max_events: int | None = None,
        max_trials: int | None = None,
        return_on_max_trials: bool = False,
        ci_confidence_level: float | None = None,
    ) -> tuple[bool, float, int, bool]: ...

    def calc_forward_probability(
        self,
        state: State,
        num_trials: int,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float: ...

    def calc_forward_probability_adaptive(
        self,
        state: State,
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]: ...

    def calc_forward_probabilities_adaptive(
        self,
        states: list[State],
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]: ...

    def place_tile(self, state: State, point: tuple[int, int], tile: int) -> float: ...

    def find_first_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None: ...

    def find_last_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None: ...

    def reconstruct_state_from_trajectory(
        self,
        trajectory: pl.DataFrame,
        up_to_index: int,
        config: CriticalStateConfig = ...,
        filter_trajectory: bool = True,
    ) -> State: ...

class KBlock:
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

    def calc_dimers(self) -> List[DimerInfo]:
        """
        Calculate information about the dimers the system is able to form.

        Returns
        -------
        List[DimerInfo]
        """

    def calc_mismatch_locations(self, state: State | FFSStateRef) -> NDArray[np.uint64]:
        """
        Calculate the locations of mismatches in the state.

        This returns a copy of the canvas, with the values set to 0 if there is no mismatch
        in the location, and > 0, in a model defined way, if there is at least one mismatch.
        Most models use v = 8*N + 4*E + 2*S + W, where N, E, S, W are the four directions.
        Thus, a tile with mismatches to the E and W would have v = 4+2 = 6.

        Parameters
        ----------
        state : State or FFSStateRef
           The state to calculate mismatches for.

        Returns
        -------
        ndarray
          An array of the same shape as the state's canvas, with the values set as described above.
        """

    def calc_mismatches(self, state: State | FFSStateRef) -> int:
        """
        Calculate the number of mismatches in a state.

        Parameters
        ----------
        state : State or FFSStateRef
          The state to calculate mismatches for.

        Returns
        -------
        int
         The number of mismatches.

        See also
        --------
        calc_mismatch_locations
          Calculate the location and direction of mismatches, not jus the number.
        """

    @overload
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
    ) -> EvolveOutcome: ...

    @overload
    def evolve(
        self,
        state: Sequence[State],
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
    ) -> List[EvolveOutcome]: ...

    @overload
    def evolve(
        self,
        state: State | Sequence[State],
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
    ) -> EvolveOutcome | List[EvolveOutcome]:
        """
        Evolve a state (or states), with some bounds on the simulation.

        If evolving multiple states, the bounds are applied per-state.

        Parameters
        ----------
        state : State or Sequence[State]
          The state or states to evolve.
        for_events : int, optional
          Stop evolving each state after this many events.
        total_events : int, optional
          Stop evelving each state when the state's total number of events (including
          previous events) reaches this.
        for_time : float, optional
          Stop evolving each state after this many seconds of simulated time.
        total_time : float, optional
          Stop evolving each state when the state's total time (including previous steps)
          reaches this.
        size_min : int, optional
          Stop evolving each state when the state's number of tiles is less than or equal to this.
        size_max : int, optional
          Stop evolving each state when the state's number of tiles is greater than or equal to this.
        for_wall_time : float, optional
          Stop evolving each state after this many seconds of wall time.
        require_strong_bound : bool
          Require that the stopping conditions are strong, i.e., they are guaranteed to be eventually
          satisfied under normal conditions.
        show_window : bool
          Show a graphical UI window while evolving (requires ui feature, and a single state).
        start_window_paused : bool
          If show_window is True, start the GUI window in a paused state. Defaults to True.
        parallel : bool
          Use multiple threads.
        initial_timescale : float, optional
          If show_window is True, set the initial timescale (sim_time/real_time) in the GUI. None means unlimited.
        initial_max_events_per_sec : int, optional
          If show_window is True, set the initial max events per second limit in the GUI. None means unlimited.

        Returns
        -------
        EvolveOutcome or List[EvolveOutcome]
         The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
        """

    def get_param(self, param_name: str) -> Any: ...

    def print_debug(self) -> None: ...

    @staticmethod
    def read_json(filename: str) -> Self:
        """
        Read a system from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to read from.
        """

    def run_ffs(self, config: FFSRunConfig = ..., **kwargs: Any) -> FFSRunResult:
        """
        Run FFS.

        Parameters
        ----------
        config : FFSRunConfig
         The configuration for the FFS run.
        **kwargs
          FFSRunConfig parameters as keyword arguments.

        Returns
        -------
        FFSRunResult
         The result of the FFS run.
        """

    def set_param(self, param_name: str, value: Any) -> NeededUpdate:
        """
        Set a system parameter.

        Parameters
        ----------
        param_name : str
            The name of the parameter to set.
        value : Any
            The value to set the parameter to.

        Returns
        -------
        NeededUpdate
            The type of state update needed.  This can be passed to
           `update_state` to update the state.
        """

    def tile_color(self, tile_number: int) -> list[int]:
        """
        Given a tile number, return the color of the tile.

        Parameters
        ----------
        tile_number : int
         The tile number.

        Returns
        -------
        list[int]
          The color of the tile, as a list of 4 integers (RGBA).
        """

    def tile_number_from_name(self, tile_name: str) -> int | None:
        """
        Given a tile name, return the tile number.

        Parameters
        ----------
        tile_name : str
          The name of the tile.

        Returns
        -------
        int | None
         The tile number, or None if not found.
        """

    def update_all(self, state: State, needed: NeededUpdate = ...) -> None: ...

    def update_state(self, state: State, needed: NeededUpdate = ...) -> None:
        """
        Recalculate a state's rates.

        This is usually needed when a parameter of the system has
        been changed.

        Parameters
        ----------
        state : State
          The state to update.
        needed : NeededUpdate, optional
          The type of update needed.  If not provided, all locations
          will be recalculated.
        """

    def setup_state(self, state: State) -> None: ...

    def write_json(self, filename: str) -> None:
        """
        Write the system to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        """

    def color_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.uint8]: ...

    def name_canvas(
        self, state: State | FFSStateRef | NDArray[np.uint]
    ) -> NDArray[np.str_]: ...

    def calc_committer(
        self,
        state: State,
        cutoff_size: int,
        num_trials: int,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float: ...

    def calc_committer_adaptive(
        self,
        state: State,
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]: ...

    def calc_committers_adaptive(
        self,
        states: list[State],
        cutoff_size: int,
        conf_interval_margin: float,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]: ...

    def calc_committer_threshold_test(
        self,
        state: State,
        cutoff_size: int,
        threshold: float,
        confidence_level: float,
        max_time: float | None = None,
        max_events: int | None = None,
        max_trials: int | None = None,
        return_on_max_trials: bool = False,
        ci_confidence_level: float | None = None,
    ) -> tuple[bool, float, int, bool]: ...

    def calc_forward_probability(
        self,
        state: State,
        num_trials: int,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> float: ...

    def calc_forward_probability_adaptive(
        self,
        state: State,
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[float, int]: ...

    def calc_forward_probabilities_adaptive(
        self,
        states: list[State],
        conf_interval_margin: float,
        forward_step: int = 1,
        max_time: float | None = None,
        max_events: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uintp]]: ...

    def place_tile(self, state: State, point: tuple[int, int], tile: int) -> float: ...

    def find_first_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None: ...

    def find_last_critical_state(
        self,
        end_state: State,
        config: CriticalStateConfig = ...,
    ) -> CriticalStateResult | None: ...

    def reconstruct_state_from_trajectory(
        self,
        trajectory: pl.DataFrame,
        up_to_index: int,
        config: CriticalStateConfig = ...,
        filter_trajectory: bool = True,
    ) -> State: ...


System: TypeAlias = ATAM | KTAM | OldKTAM | KBlock | SDC

class State:
    def __init__(
        self,
        shape: tuple[int, int],
        kind: str = "Square",
        tracking: str = "None",
        n_tile_types: int | None = None,
    ) -> None: ...

    @staticmethod
    def from_array(
        array: NDArray[np.uint],
        kind: str = "Square",
        tracking: str = "None",
        n_tile_types: int | None = None,
    ) -> State:
        """Create a state from an existing numpy array."""

    @property
    def canvas_view(self) -> NDArray[np.uint]:
        """A view of the state's canvas.  This is fast but unsafe."""

    def canvas_copy(self) -> ndarray:
        """A copy of the state's canvas.  This is safe, but can't be modified and is slower than `canvas_view`."""

    def rate_array(self) -> NDArray[np.float64]:
        """Return a cloned copy of an array with the total possible next event rate for each point in the canvas."""

    @property
    def total_rate(self) -> float:
        """The total rate of possible next events for the state."""

    def print_debug(self) -> None: ...
    def rate_at_point(self, point: tuple[int, int]) -> float: ...
    @staticmethod
    def read_json(filename: str) -> State: ...
    def tracking_copy(self) -> ndarray: ...
    def write_json(self, filename: str) -> None: ...
    def copy(self) -> Self:
        """
        Create a copy of the state.

        This creates a complete clone of the state, including all canvas data,
        tracking information, and simulation state (time, events, etc.).

        Returns
        -------
        State
            A new State object that is a copy of this state.

        Examples
        --------
        >>> original_state = State((10, 10))
        >>> copied_state = original_state.copy()
        >>> # The copied state is independent of the original
        >>> assert copied_state.time == original_state.time
        >>> assert copied_state.total_events == original_state.total_events
        """
    @property
    def n_tiles(self) -> int:
        """The number of tiles in the state."""

    @property
    def ntiles(self) -> int:
        """The number of tiles in the state (deprecated, use `n_tiles` instead)."""

    @property
    def total_events(self) -> int:
        """The total number of events that have occurred in the state."""

    @property
    def time(self) -> float:
        """The total time the state has simulated, in seconds."""

    @property
    def tile_counts(self) -> NDArray[np.uint32]:
        """Counts of each tile type in the state."""

    def __getstate__(self) -> bytes:
        """Serialize state for pickling."""

    def __setstate__(self, state: bytes) -> None:
        """Deserialize state from pickle data."""

    def __getnewargs__(self) -> tuple[tuple[int, int]]:
        """Return arguments for __new__ during unpickling."""

    def replay(self, up_to_event: int | None = None) -> Self:
        """
        Replay the events from a MovieTracker up to a given event ID.

        This reconstructs the state by replaying all events from the MovieTracker.
        The state must have been created with Movie tracking enabled.

        Parameters
        ----------
        up_to_event : int, optional
            The event ID up to which to replay (inclusive). If not provided,
            all events are replayed.

        Returns
        -------
        State
            A new State with the events replayed. The returned state has no
            tracker and no rates calculated.

        Raises
        ------
        ValueError
            If the state does not have a MovieTracker.

        Examples
        --------
        >>> # Create a state with movie tracking and evolve it
        >>> state = ts.create_state(tracking="Movie")
        >>> sys.evolve(state, for_events=100)
        >>> # Replay to get state at event 50
        >>> replayed = state.replay(up_to_event=50)
        """

    def replay_inplace(
        self,
        coords: list[tuple[int, int]],
        new_tiles: list[int],
        event_ids: list[int],
        up_to_event_id: int,
        n_tiles: list[int] | None = None,
        total_time: list[float] | None = None,
        energy: list[float] | None = None,
    ) -> None:
        """
        Replay events in-place on this state from external event data.

        This modifies the state's canvas by applying the events from the provided
        coordinate and tile arrays. Unlike `replay()`, this method takes external
        event data rather than using a MovieTracker.

        Parameters
        ----------
        coords : list[tuple[int, int]]
            List of (row, col) coordinates for each event.
        new_tiles : list[int]
            List of tile values for each event.
        event_ids : list[int]
            List of event IDs for each event.
        up_to_event_id : int
            The event ID up to which to replay (inclusive).
        n_tiles : list[int], optional
            List of tile counts at each event. If provided, sets the state's
            n_tiles to the value at the last replayed event.
        total_time : list[float], optional
            List of total simulation times at each event. If provided, sets
            the state's time to the value at the last replayed event.
        energy : list[float], optional
            List of energies at each event. If provided, sets the state's
            energy to the value at the last replayed event.

        Raises
        ------
        ValueError
            If there is an error during replay.

        Examples
        --------
        >>> state = State((10, 10))
        >>> coords = [(1, 1), (2, 2)]
        >>> new_tiles = [1, 2]
        >>> event_ids = [0, 1]
        >>> state.replay_inplace(coords, new_tiles, event_ids, 1)
        """

class TileSet:
    def __init__(self, **kwargs: Any) -> None: ...
    def create_state(self, system: System | None = None) -> State: ...
    def create_state_from_canvas(self, canvas: NDArray[np.uint]) -> State: ...
    def create_state_empty(self, system: System | None = None) -> State: ...
    def create_system(self) -> System: ...
    def create_system_and_state(self) -> tuple[System, State]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Creates a TileSet from a dict by exporting to json, then parsing the json.
        FIXME: implement this without the json trip.
        """

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Parses a file (JSON, YAML, etc) into a TileSet"""

    @classmethod
    def from_json(cls, data: str) -> Self:
        """Parses a JSON string into a TileSet."""

    def run_ffs(self, config: FFSRunConfig = ..., **kwargs: Any) -> FFSRunResult:
        """Runs FFS."""

    def run_window(self) -> State:
        """
        Creates a simulation, and runs it in a UI.  Returns the State when
        finished.
        """

class TileShape:
    """Shape of a tile (Single, Horizontal, or Vertical)."""
    Single: TileShape
    Horizontal: TileShape
    Vertical: TileShape

class Orientation:
    """Orientation of a dimer (NS = North-South, WE = West-East)."""
    NS: Orientation
    WE: Orientation

class DimerInfo:
    """Information about a dimer that can form in the system."""
    @property
    def t1(self) -> int:
        """First tile number in the dimer."""
    @t1.setter
    def t1(self, value: int) -> None: ...

    @property
    def t2(self) -> int:
        """Second tile number in the dimer."""
    @t2.setter
    def t2(self, value: int) -> None: ...

    @property
    def orientation(self) -> Orientation:
        """Orientation of the dimer (NS or WE)."""
    @orientation.setter
    def orientation(self, value: Orientation) -> None: ...

    @property
    def formation_rate(self) -> float:
        """Formation rate in M/s."""
    @formation_rate.setter
    def formation_rate(self, value: float) -> None: ...

    @property
    def equilibrium_conc(self) -> float:
        """Equilibrium concentration in M."""
    @equilibrium_conc.setter
    def equilibrium_conc(self, value: float) -> None: ...

class NeededUpdate: ...

class AnnealProtocol:
    """Protocol for running annealing simulations."""
    def __init__(
        self,
        from_tmp: float,
        to_tmp: float,
        initial_hold: float,
        final_hold: float,
        delta_time: float,
        scaffold_count: int,
        seconds_per_step: float,
    ) -> None: ...

    def run_one_system(self, sdc: SDC) -> AnnealOutput | None: ...
    def run_many_systems(self, sdcs: list[SDC]) -> list[AnnealOutput | None]: ...

class AnnealOutput: ...

def string_dna_dg_ds(dna_sequence: str) -> tuple[float, float]: ...
def get_color(color_name: str) -> list[int]: ...
def get_color_or_random(color_name: str | None) -> list[int]: ...
def loop_penalty(loop_length: int) -> float: ...
