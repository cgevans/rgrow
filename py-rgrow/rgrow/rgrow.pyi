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
    def tile_colors(self) -> NDArray[np.uint]: ...
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
        parallel: bool = True,
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
        parallel: bool = True,
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
        parallel: bool = True,
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
        parallel : bool
          Use multiple threads.

        Returns
        -------
        EvolveOutcome or List[EvolveOutcome]
         The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
        """

    def get_param(self, param_name): ...
    def print_debug(self): ...
    @staticmethod
    def read_json(filename: str) -> None:
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

    def tile_number_from_name(self, tile_name: str) -> int:
        """
        Given a tile name, return the tile number.

        Parameters
        ----------
        tile_name : str
          The name of the tile.

        Returns
        -------
        int
         The tile number.
        """

    def update_all(self, state, needed=...): ...
    def update_state(self, state: State, needed: NeededUpdate | None = ...) -> None:
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

class SDC:
    @property
    def tile_names(self) -> list[str]: ...
    @property
    def tile_colors(self) -> NDArray[np.uint]: ...
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
        parallel: bool = True,
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
        parallel: bool = True,
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
        parallel: bool = True,
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
        parallel : bool
          Use multiple threads.

        Returns
        -------
        EvolveOutcome or List[EvolveOutcome]
         The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
        """

    def get_param(self, param_name): ...
    def print_debug(self): ...
    @staticmethod
    def read_json(filename: str) -> None:
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

    def tile_number_from_name(self, tile_name: str) -> int:
        """
        Given a tile name, return the tile number.

        Parameters
        ----------
        tile_name : str
          The name of the tile.

        Returns
        -------
        int
         The tile number.
        """

    def update_all(self, state, needed=...): ...
    def update_state(self, state: State, needed: NeededUpdate | None = ...) -> None:
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

class EvolveBounds:
    def __init__(self, for_time: float | None = None): ...
    def is_strongly_bounded(self) -> bool: ...
    def is_weakly_bounded(self) -> bool:
        """
        Will the EvolveBounds actually bound anything, or is it just null, such that the simulation will continue
        until a ZeroRate or an error?  Note that this includes weak bounds (size minimum and maximum) that may
        never be reached.
        """

class EvolveOutcome: ...

class FFSLevelRef:
    def get_state(self, i): ...
    def has_stored_states(self): ...

class FFSRunConfig:
    """Configuration options for FFS."""

class FFSRunResult:
    def configs_dataframe(self): ...
    def into_resdf(self): ...
    def surfaces_dataframe(self): ...
    def write_files(self, prefix): ...

class FFSRunResultDF:
    def configs_dataframe(self): ...
    def read_files(prefix) -> Self:
        """
        Read dataframes and result data from files.

        Returns
        -------
        Self
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

class FFSStateRef:
    def canvas_copy(self) -> None:
        """A copy of the state's canvas.  This is safe, but can't be modified and is slower than `canvas_view`."""

    def clone_state(self): ...
    @property
    def canvas_view(self) -> NDArray[np.uint]: ...
    def n_tiles(self) -> int: ...
    def time(self) -> float: ...
    def total_events(self) -> int: ...
    def tracking_copy(self) -> Any: ...

class KTAM:
    @property
    def tile_names(self) -> list[str]: ...
    @property
    def tile_colors(self) -> NDArray[np.uint]: ...
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
        parallel: bool = True,
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
        parallel: bool = True,
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
        parallel: bool = True,
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
        parallel : bool
          Use multiple threads.

        Returns
        -------
        EvolveOutcome or List[EvolveOutcome]
         The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
        """

    def from_tileset(tileset): ...
    def get_param(self, param_name): ...
    def print_debug(self): ...
    @staticmethod
    def read_json(filename: str) -> None:
        """
        Read a system from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to read from.
        """

    def run_ffs(self, config: FFSRunConfig = ..., **kwargs) -> FFSRunResult:
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

    def tile_number_from_name(self, tile_name: str) -> int:
        """
        Given a tile name, return the tile number.

        Parameters
        ----------
        tile_name : str
          The name of the tile.

        Returns
        -------
        int
         The tile number.
        """

    def update_all(self, state, needed=...): ...
    def update_state(self, state: State, needed: NeededUpdate | None = ...) -> None:
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

class OldKTAM:
    @property
    def tile_names(self) -> list[str]: ...
    @property
    def tile_colors(self) -> NDArray[np.uint]: ...
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
        parallel: bool = True,
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
        parallel: bool = True,
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
        parallel: bool = True,
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
        parallel : bool
          Use multiple threads.

        Returns
        -------
        EvolveOutcome or List[EvolveOutcome]
         The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
        """

    def get_param(self, param_name): ...
    def print_debug(self): ...
    @staticmethod
    def read_json(filename: str) -> None:
        """
        Read a system from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to read from.
        """

    def run_ffs(self, config: FFSRunConfig = ..., **kwargs) -> FFSRunResult:
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

    def tile_number_from_name(self, tile_name: str) -> int:
        """
        Given a tile name, return the tile number.

        Parameters
        ----------
        tile_name : str
          The name of the tile.

        Returns
        -------
        int
         The tile number.
        """

    def update_all(self, state, needed=...): ...
    def update_state(self, state: State, needed: NeededUpdate | None = ...) -> None:
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

    def write_json(self, filename: str) -> None:
        """
        Write the system to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        """

System: TypeAlias = ATAM | KTAM | OldKTAM

class State:
    def __init__(
        self,
        shape: tuple[int, int],
        kind: str = "Square",
        tracking: str = "None",
        n_tile_types: int | None = None,
    ): ...
    @property
    def canvas_view(self) -> NDArray[np.uint]:
        """A view of the state's canvas.  This is fast but unsafe."""

    def canvas_copy(self) -> ndarray:
        """A copy of the state's canvas.  This is safe, but can't be modified and is slower than `canvas_view`."""

    def print_debug(self) -> None: ...
    def rate_at_point(self, point: tuple[int, int]) -> float: ...
    @staticmethod
    def read_json(filename: str) -> State: ...
    def tracking_copy(self) -> ndarray: ...
    def write_json(self, filename: str) -> None: ...
    @property
    def ntiles(self) -> int: ...
    @property
    def total_events(self) -> int: ...
    @property
    def time(self) -> float: ...

class TileSet:
    def __init__(self, **kwargs: Any): ...
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
    def from_file(cls, path) -> Self:
        """Parses a file (JSON, YAML, etc) into a TileSet"""

    @classmethod
    def from_json(cls, data) -> Self:
        """Parses a JSON string into a TileSet."""

    def run_ffs(self, config: FFSRunConfig = ..., **kwargs: Any) -> FFSRunResult:
        """Runs FFS."""

    def run_window(self) -> EvolveOutcome:
        """
        Creates a simulation, and runs it in a UI.  Returns the :any:`Simulation` when
        finished.
        """

class TileShape: ...
class DimerInfo: ...
class NeededUpdate: ...

def string_dna_dg_ds(dna_sequence: str) -> tuple[float, float]: ...
