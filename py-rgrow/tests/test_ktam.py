from typing import cast
from rgrow import EvolveOutcome, Tile, TileSet, Bond  # noqa: F841
import pytest  # noqa: F401
from rgrow import KTAM, State

def test_ktam_growth():
    tube_ts = TileSet(
        [
            Tile(["a","a","b","b"],),
            Tile(["b","b","a","a"]),
        ],
        [Bond("a", 1), Bond("b", 1)],
        canvas_type="tube",
        size=(8, 256),
        alpha=-7.1,
        gse=5.05,
        gmc=10.0
    )

    sys, state = cast(tuple[KTAM, State], tube_ts.create_system_and_state())
    
    # We'll start with some tiles:
    state.canvas_view[::2, 5:50] = 1
    state.canvas_view[1::2, 5:50] = 2

    sys.update_all(state)

    # Should have no mismatches:
    assert sys.calc_mismatches(state) == 0

    start_n = state.n_tiles

    # We should grow in these conditions:
    out = sys.evolve(state, for_events=100_000, size_min=0, size_max=2*start_n)

    # We should have reached the max size:
    assert out == EvolveOutcome.ReachedSizeMax

def test_ktam_melt():
    tube_ts = TileSet(
        [
            Tile(["a","a","b","b"],),
            Tile(["b","b","a","a"]),
        ],
        [Bond("a", 1), Bond("b", 1)],
        canvas_type="tube",
        size=(8, 256),
        alpha=-7.1,
        gse=4.95,
        gmc=10.0
    )

    sys, state = cast(tuple[KTAM, State], tube_ts.create_system_and_state())
    
    # We'll start with some tiles:
    state.canvas_view[::2, 5:50] = 1
    state.canvas_view[1::2, 5:50] = 2

    sys.update_all(state)

    # Should have no mismatches:
    assert sys.calc_mismatches(state) == 0
    
    start_n = state.n_tiles

    # We should melt in these conditions:
    out = sys.evolve(state, for_events=100_000, size_min=0, size_max=2*start_n)

    # We should have reached the max size:
    assert out == EvolveOutcome.ReachedSizeMin

def test_ktam_equilibrium():
    tube_ts = TileSet(
        [
            Tile(["a","a","b","b"],),
            Tile(["b","b","a","a"]),
        ],
        [Bond("a", 1), Bond("b", 1)],
        canvas_type="tube",
        size=(8, 256),
        alpha=-7.1,
        gse=5.0,
        gmc=10.0
    )

    sys, state = cast(tuple[KTAM, State], tube_ts.create_system_and_state())
    
    # We'll start with some tiles:
    state.canvas_view[::2, 5:50] = 1
    state.canvas_view[1::2, 5:50] = 2

    sys.update_all(state)

    # Should have no mismatches:
    assert sys.calc_mismatches(state) == 0
    
    start_n = state.n_tiles

    # We should melt in these conditions:
    out = sys.evolve(state, for_events=100_000, size_min=0, size_max=2*start_n)

    # We should run out of events, hopefully.
    assert out == EvolveOutcome.ReachedEventsMax