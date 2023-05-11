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

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt
    import matplotlib.colors


def _plot_state(
    self: Simulation, state_index: int = 0, ax: "int | plt.Axes" = None
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


def _tile_cmap(self: Simulation) -> "matplotlib.colors.ListedColormap":
    """Returns a matplotlib colormap for tile numbers."""
    from matplotlib.colors import ListedColormap
    import numpy as np

    return ListedColormap(
        np.array(self.tile_colors) / 255,
        name="tile_cmap",
    )


Simulation.plot_state = _plot_state  # type: ignore
Simulation.tile_cmap = _tile_cmap  # type: ignore
