from pathlib import Path
from .anneal import AnnealOutputs
from .reporter_methods import ReportingMethod

import matplotlib.pyplot as plt

MIN = 60
HOUR = MIN * 60


class Graphing:
    def __init__(self, title: str):
        self.title = title
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        # Without this line, notebooks will display an empty graph when __init__ is called.
        plt.close(self.fig)
        self._times = None
        self._temps = None

    def add_line(self, anneal_output: AnnealOutputs, method: ReportingMethod, *args, **kwargs):
        """
        Adds a line to the graph using a reporting function to extract data.

        Parameters
        ----------
        Todo
        """
        measurement = method.reporter_method(anneal_output)
        times, temps = anneal_output.anneal.gen_arrays()
        times_hours = times / HOUR
        
        if self._times is None:
            self._times = times_hours
            self._temps = temps
        elif not (len(times_hours) == len(self._times) and all(times_hours == self._times)):
            raise ValueError("Time arrays do not match across added lines.")

        self.ax1.plot(times_hours, measurement, *args, **kwargs)
        self.ax1.set_ylabel(method.desc)
        self.ax1.set_ylim(0.0, 1.1)

    def end(self, save_path: str | None = None):
        """
        Finalizes and renders or saves the plot.
        """
        self.ax1.set_title(self.title)
        self.ax1.set_xlabel("Time (hours)")
        self.ax1.legend()
        self.ax1.grid(True)

        # Plot temperature on right-hand axis
        if self._temps is not None:
            self.ax2.plot(self._times, self._temps, "k--", label="Temperature (°C)")
            self.ax2.set_ylabel("Temperature (°C)")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(save_path)
            print(f"[INFO] Saved plot to: {save_path}")
        else:
            self.fig.tight_layout()
            return self.fig
