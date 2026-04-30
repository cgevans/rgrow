"""Python facade for the SDC2D model.

Provides plain dataclasses (`SDC2DStrand`, `SDC2DParams`) that the Rust
extension extracts via attribute access. The actual model class is the
Rust `SDC2D` re-exported from `rgrow.rgrow`.

`SDC2D` exposes exact finite-grid thermodynamics methods including
``mfe_config()``, ``state_g()``, ``log_partition_function()``,
``partition_function()``, ``log_partial_partition_function()``,
``partial_partition_function()``, ``probability_of_state()``, and
``probability_of_constrained_configurations()``. These calculations are exact
but exponential in the smaller scaffold dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .rgrow import SDC2D

__all__ = ["SDC2DStrand", "SDC2DParams", "SDC2D"]


@dataclass
class SDC2DStrand:
    """A strand with five glues (W, N, E, S, Bottom)."""

    concentration: float = 1e-6
    west_glue: str | None = None
    north_glue: str | None = None
    east_glue: str | None = None
    south_glue: str | None = None
    bottom_glue: str | None = None
    name: str | None = None
    color: str | None = None


@dataclass
class SDC2DParams:
    """Parameters for an SDC2D system.

    Attributes
    ----------
    strands
        Strands available in solution.
    scaffold
        2D layout of glue names. ``scaffold[r][c]`` is the glue at row ``r``,
        column ``c``; use ``None`` for "no scaffold glue here". Border cells
        outside the canvas-usable region (the 2-cell border that ``Square``
        canvases enforce) should be ``None``.
    scaffold_concentration
        Effective molar concentration of the scaffold.
    glue_dg_s
        Per-glue or per-pair `(ΔG_37, ΔS)` thermodynamics, or a DNA sequence
        string. Single namespace shared by all four lateral edges and the
        bottom edge.
    k_f
        Forward rate constant (1/(M·s)).
    temperature
        Temperature in Celsius.
    seed
        Anchor strands as ``(row, col, strand_name)`` tuples; pinned strands
        cannot detach.
    """

    strands: list[SDC2DStrand] = field(default_factory=list)
    scaffold: list[list[str | None]] = field(default_factory=list)
    scaffold_concentration: float = 1e-9
    glue_dg_s: Mapping[
        str | tuple[str, str], tuple[float, float] | str
    ] = field(default_factory=dict)
    k_f: float = 1e6
    temperature: float = 37.0
    seed: list[tuple[int, int, str]] = field(default_factory=list)
