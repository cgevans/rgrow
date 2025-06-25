import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING
import pickle
from platformdirs import user_data_dir
from pathlib import Path
import sys

if TYPE_CHECKING:
    from .sdc import SDC
    from rgrow import State

MIN = 60
HOUR = MIN * 60


@dataclass
class Anneal:
    """
    An anneal protocol.

    Attributes:
    ----------
    initial_hold : float
        How long to hold the system for before changing temperature (in seconds)
    final_hold : float
        How long to hold the system for once the temperature is finished changing (in seconds)
    delta_time : float
        The duration of time during which the temperature will be changing (in seconds)
    initial_temperature : float
        Temperature of the system before anneal starts (in degrees C)
    final_temperature : float
        Target temperature, it will be reached at the end of the anneal (in degrees C)
    scaffold_count : int
        Number of scaffolds to simulate, the higher, the more statistically significant, but the longer the anneal will
        take to finish running
    timestep : float
        Simulated time cannot be continuous. How big do you want each time jump to be ? The smaller, the more accurate
        the system will be, but it will take longer.
    temperature_adjustment : float
        How much to adjust the temperature to correct for a model temperature offset.
    """

    initial_hold: float
    initial_tmp: float
    delta_time: float
    final_tmp: float
    final_hold: float
    scaffold_count: int = 100
    timestep: float = 2.0
    temperature_adjustment: float = 8.0

    @property
    def adjusted_initial_tmp(self):
        return self.initial_tmp + self.temperature_adjustment

    @property
    def adjusted_final_tmp(self):
        return self.final_tmp + self.temperature_adjustment

    @staticmethod
    def standard_long_anneal(from_tmp=80, final_tmp=20, scaffold_count=100):
        """
        Standard anneal with:
            Rest for 10 minutes, then change temperatures linearly
            for 3 hours, then rest for 45 minutes.

            The system will go from the initial temperature (default 80+8),
            to the final temperature (default 20+8) over the 3 hours.
        """
        # error_delta = 8
        return Anneal(10 * MIN, from_tmp, 3 * HOUR, final_tmp, 45 * MIN, scaffold_count)

    def gen_arrays(self):
        """
        Generate the time and the temperature arrays

        Returns:
        -------
        times : np.ndarray
            An array of times
        temperatures : np.ndarray
            An array of temperatures
        """
        steps_per_sec = 1 / self.timestep
        number_of_steps = int(self.delta_time * steps_per_sec)

        delta_temperatures = np.linspace(
            self.adjusted_initial_tmp, self.adjusted_final_tmp, int(
                number_of_steps + 1)
        )
        initial_temp = np.repeat(
            self.adjusted_initial_tmp, int(
                self.initial_hold * steps_per_sec) - 1
        )
        ending_temp = np.repeat(
            self.adjusted_final_tmp, int(self.final_hold * steps_per_sec)
        )
        temperatures = np.concatenate(
            [initial_temp, delta_temperatures, ending_temp])

        total_time = self.initial_hold + self.final_hold + self.delta_time
        times = np.arange(self.timestep, total_time +
                          self.timestep, self.timestep)

        return times, temperatures


@dataclass
class AnnealOutputs:
    """
    Stores the result of an anneal simulation on an SDC system.

    Attributes:
    ----------
    system : SDC
        The SDC system instance that was simulated.
    canvas_arr : np.ndarray
        3D array capturing scaffold states at each time point.

        Each `canvas_arr[t][n][i]` contains the ID of the strand (or 0 if none)
        bound at scaffold position `i` at time step `t`, for scaffold `n`.
        Note that the first two, and the last two indices of each scaffold will
        always be empty. That is, the position A in the scaffold has index 2, not
        0.
    anneal : Anneal
        The annealing protocol that was executed.
    state : State
        Final simulation state (e.g., to resume or analyze thermodynamic properties).
    """
    system: "SDC"
    canvas_arr: "np.ndarray"
    anneal: "Anneal"
    state: "State"

    def save_data(self, file_name: str):
        try:
            app_dir = Path(user_data_dir("rgrow")) / "sdc"
            app_dir.mkdir(parents=True, exist_ok=True)
            file_path = app_dir / file_name
            data = {
                "canvas_arr": self.canvas_arr,
                "anneal": self.anneal,
                "sdc_params": self.system.params,
                "sdc_name": self.system.name,
            }
            with file_path.open("wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[ERROR] Failed to write file: {e}", file=sys.stderr)

    @staticmethod
    def load_data(file_name: str) -> "AnnealOutputs":
        """
        Loads a previously saved simulation result, and reconstructs the system and state.
        """
        from .sdc import SDC

        try:
            file_path = Path(user_data_dir("rgrow")) / "sdc" / file_name
            with file_path.open("rb") as f:
                data = pickle.load(f)

            sdc = SDC(data["sdc_params"], data["sdc_name"])
            return AnnealOutputs(
                system=sdc,
                canvas_arr=data["canvas_arr"],
                anneal=data["anneal"],
                state=None
            )
        except Exception as e:
            print(f"[ERROR] Failed to load file '{file_name}': {e}", file=sys.stderr)
