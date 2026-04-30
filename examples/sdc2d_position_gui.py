"""Run a small position-addressed SDC2DSquare system in the rgrow GUI.

The model uses an 8x8 scaffold. Every scaffold position has two position-
specific tile types, a 0 tile and a 1 tile, except the bottom-left site. The
bottom-left site is either a pinned seed tile that presents 0-tile lateral
bonds, or an unpinned 0 tile with a strong scaffold bond.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass

import rgrow as rg


N = 8
CONC = 1e-6
BOTTOM_LEFT = (N - 1, 0)


def _load_sdc2d_api():
    try:
        from rgrow.sdc2d import SDC2DSquare, SDC2DParams, SDC2DStrand

        return SDC2DSquare, SDC2DParams, SDC2DStrand
    except ImportError:
        from rgrow.rgrow import SDC2DSquare

        SDC2DParams = collections.namedtuple(
            "SDC2DParams",
            [
                "strands",
                "scaffold",
                "scaffold_concentration",
                "glue_dg37_ds",
                "k_f",
                "temperature",
                "seed",
            ],
        )
        SDC2DStrand = collections.namedtuple(
            "SDC2DStrand",
            [
                "name",
                "color",
                "concentration",
                "west_glue",
                "north_glue",
                "east_glue",
                "south_glue",
                "bottom_glue",
            ],
        )
        return SDC2DSquare, SDC2DParams, SDC2DStrand


SDC2DSquare, SDC2DParams, SDC2DStrand = _load_sdc2d_api()


@dataclass(frozen=True)
class GlueSet:
    west: str | None
    north: str | None
    east: str | None
    south: str | None


def position_glue(row: int, col: int) -> str:
    return f"pos_{row}_{col}"


def lateral_glues(row: int, col: int, bit: int) -> GlueSet:
    west = f"h_{row}_{col - 1}_{bit}*" if col > 0 else None
    east = f"h_{row}_{col}_{bit}" if col < N - 1 else None
    north = f"v_{row - 1}_{col}_{bit}*" if row > 0 else None
    south = f"v_{row}_{col}_{bit}" if row < N - 1 else None
    return GlueSet(west=west, north=north, east=east, south=south)


def make_strand(row: int, col: int, bit: int, *, name: str | None = None) -> object:
    glues = lateral_glues(row, col, bit)
    color = "#2f6fdd" if bit == 0 else "#d69225"
    return SDC2DStrand(
        name=name or f"t_{row}_{col}_{bit}",
        color=color,
        concentration=CONC,
        west_glue=glues.west,
        north_glue=glues.north,
        east_glue=glues.east,
        south_glue=glues.south,
        bottom_glue=position_glue(row, col),
    )


def build_system(*, gc: float, gs: float, gi: float, seed_mode: bool, temperature: float):
    strands = []
    seed = []
    glue_dg37_ds: dict[str, tuple[float, float]] = {}

    for row in range(N):
        for col in range(N):
            is_bottom_left = (row, col) == BOTTOM_LEFT
            if is_bottom_left and seed_mode:
                seed_name = f"seed_{row}_{col}_0"
                strands.append(make_strand(row, col, 0, name=seed_name))
                seed.append((row, col, seed_name))
            elif is_bottom_left:
                strands.append(make_strand(row, col, 0))
            else:
                strands.append(make_strand(row, col, 0))
                strands.append(make_strand(row, col, 1))

            scaffold_energy = gs if seed_mode or not is_bottom_left else gi
            glue_dg37_ds[position_glue(row, col)] = (scaffold_energy, 0.0)

    for row in range(N):
        for col in range(N - 1):
            for bit in (0, 1):
                glue_dg37_ds[f"h_{row}_{col}_{bit}"] = (gc, 0.0)
    for row in range(N - 1):
        for col in range(N):
            for bit in (0, 1):
                glue_dg37_ds[f"v_{row}_{col}_{bit}"] = (gc, 0.0)

    scaffold = [[f"{position_glue(row, col)}*" for col in range(N)] for row in range(N)]
    params = SDC2DParams(
        strands=strands,
        scaffold=scaffold,
        scaffold_concentration=1e-100,
        glue_dg37_ds=glue_dg37_ds,
        k_f=1e6,
        temperature=temperature,
        seed=seed,
    )
    return SDC2DSquare(params), len(strands) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gc", type=float, default=-7.0, help="Adjacent tile bond energy")
    parser.add_argument("--gs", type=float, default=-8.0, help="Scaffold bond energy")
    parser.add_argument("--gi", type=float, default=-20.0, help="Anchor scaffold bond energy")
    parser.add_argument("--temperature", type=float, default=37.0)
    parser.add_argument(
        "--anchor",
        action="store_true",
        help="Use a strong unpinned anchor tile instead of a pinned seed tile",
    )
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--max-events-per-sec", type=int, default=5000)
    args = parser.parse_args()

    sys, n_tile_types = build_system(
        gc=args.gc,
        gs=args.gs,
        gi=args.gi,
        seed_mode=not args.anchor,
        temperature=args.temperature,
    )
    state = rg.State((N, N), kind="SquareCompact", tracking="None", n_tile_types=n_tile_types)
    sys.setup_state(state)
    sys.update_state(state)
    sys.evolve(
        state,
        show_window=True,
        start_window_paused=args.start_paused,
        initial_max_events_per_sec=args.max_events_per_sec,
    )


if __name__ == "__main__":
    main()
