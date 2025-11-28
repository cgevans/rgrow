from typing import cast
from rgrow import EvolveOutcome, Tile, TileSet, Bond  # noqa: F841
import pytest  # noqa: F401
from rgrow import KTAM, State

def test_ktam_growth():
    """A simple A/B checkerboard tile system, on a tube.  This should grow if two bonds are favorable, should shrink if two bonds are unfavorable,
    and should be at equilibrium at gmc=2*gse."""
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


def test_ktam_hduples():
    tube_ts = TileSet(
        [
            Tile([0,0,"a","t2","t1","a"], shape="h", name="tile1"),
            Tile(["b1","b2","b",0,0,"b"], shape="h", name="tile2"),
        ],
        [],
        glues=[("t1","b2",1), ("t2","b1",1)],
        canvas_type="square",
        size=(8, 128),
        seed=[(3,3,"tile1"),(4,4,"tile2")],
        alpha=-7.1,
        gse=5.2,
        gmc=10.0
    )

    sys, state = cast(tuple[KTAM, State], tube_ts.create_system_and_state())
    sys.update_all(state)

    # Should have no mismatches:
    assert sys.calc_mismatches(state) == 0
    
    # We should melt in these conditions:
    out = sys.evolve(state, for_events=100_000, size_min=0, size_max=100)

    # We should run out of events, hopefully.
    assert out == EvolveOutcome.ReachedSizeMax


def test_ktam_vduples():
    tube_ts = TileSet(
        [
            Tile(["a","t1","t2","a",0,0], shape="v", name="tile1"),
            Tile(["b",0,0,"b","b2","b1"], shape="v", name="tile2"),
        ],
        [],
        glues=[("t1","b2",1), ("t2","b1",1)],
        canvas_type="square",
        size=(128, 8),
        seed=[(3,3,"tile1"),(4,4,"tile2")],
        alpha=-7.1,
        gse=5.2,
        gmc=10.0
    )

    sys, state = cast(tuple[KTAM, State], tube_ts.create_system_and_state())
    sys.update_all(state)

    # Should have no mismatches:
    assert sys.calc_mismatches(state) == 0
    
    # We should melt in these conditions:
    out = sys.evolve(state, for_events=100_000, size_min=0, size_max=100)

    # We should run out of events, hopefully.
    assert out == EvolveOutcome.ReachedSizeMax

    