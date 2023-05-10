__all__ = [
    "Tile",
    "TileSet",
    "Simulation",
    "EvolveOutcome",
    "FFSLevel",
    "FFSResult",
    "FFSRunConfig",
]

from rgrow.rgrow import (
    Tile,
    TileSet,
    Simulation,
    EvolveOutcome,
    FFSLevel,
    FFSResult,
    FFSRunConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _plot_state(self: Simulation, state: int = 0, ax: "int | plt.Axes" = None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    if ax is None:
        _, ax = plt.subplots()

    v = self.canvas_view(state)
    pc = ax.pcolormesh(
        v,
        cmap=ListedColormap(np.array(self.tile_colors) / 255),
        linewidth=0.5,
        edgecolors="#ffffff",
    )
    ax.set_aspect("equal")
    ax.set_ylim(v.shape[0], 0)

    return pc


Simulation.plot_state = _plot_state
