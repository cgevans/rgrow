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


def test_ktam_fission_no_fission():
    """Test that NoFission prevents fission events from occurring.  This uses a temperature-1 system with a single tile growing
    a 1D line.  Since there is no fission, growth will be favorable from the seed."""
    tube_ts = TileSet(
        [
            Tile([0, 1, 0, 1]),
        ],
        [Bond("1", 1)],
        canvas_type="square",
        size=(8, 1024),
        alpha=-7.1,
        gse=10.2,
        gmc=10.0,
        fission="no-fission",
        seed=[(4, 2, 1)]
    )

    sys, state = cast(tuple[KTAM, State], tube_ts.create_system_and_state())
    sys.update_all(state)

    assert sys.calc_mismatches(state) == 0

    out = sys.evolve(state, for_events=100_000, size_min=0, size_max=1000)

    assert out == EvolveOutcome.ReachedSizeMax
    assert state.n_tiles > 800



def test_ktam_fission_keep_seeded():
    """Test that KeepSeeded keeps the seeded tile when fission occurs.  This cheats, using a system of two structures bound by
    a weak tile that is almost certain to detach."""
    ts = TileSet(
        [
            Tile([1, 1, 1, 1], name="tile1"),
            Tile([0, 2, 0, 2], name="tile2")
        ],
        glues=[(1,2,0.1)],
        canvas_type="square",
        size=(128, 128),
        alpha=-7.1,
        gse=10.1,
        gmc=20.0,
        fission="keep-seeded",
        seed=(4, 2, "tile1"),
    )

    sys, state = cast(tuple[KTAM, State], ts.create_system_and_state())
    state.canvas_view[3:13, 3:13] = 1
    state.canvas_view[3:13, 14:23] = 1
    state.canvas_view[8, 13] = 2
    sys.update_all(state)

    assert sys.calc_mismatches(state) == 0
    
    sys.evolve(state, for_events=100, size_min=0, size_max=1000)

    assert state.canvas_view[5, 20] == 0
    assert state.n_tiles < 150
    assert state.canvas_view[5, 10] == 1

def test_ktam_fission_keep_largest():
    """Test that KeepSeeded keeps the seeded tile when fission occurs.  This cheats, using a system of two structures bound by
    a weak tile that is almost certain to detach."""
    ts = TileSet(
        [
            Tile([1, 1, 1, 1], name="tile1"),
            Tile([0, 2, 0, 2], name="tile2")
        ],
        glues=[(1,2,0.1)],
        canvas_type="square",
        size=(128, 128),
        alpha=-7.1,
        gse=5.1,
        gmc=10.0,
        fission="keep-largest",
        seed=(4, 2, "tile1"),
    )

    sys, state = cast(tuple[KTAM, State], ts.create_system_and_state())
    state.canvas_view[3:13, 3:13] = 1
    state.canvas_view[3:13, 14:64] = 1
    state.canvas_view[8, 13] = 2
    sys.update_all(state)

    assert sys.calc_mismatches(state) == 0

    assert state.n_tiles > 600
    sys.evolve(state, for_events=100, size_min=0, size_max=1000)

    assert state.canvas_view[5, 20] == 1
    assert state.n_tiles < 550
    assert state.canvas_view[5, 10] == 0

def test_ktam_fission_keep_weighted():
    """Test that KeepWeighted uses weighted selection when fission occurs."""
    """Test that KeepSeeded keeps the seeded tile when fission occurs.  This cheats, using a system of two structures bound by
    a weak tile that is almost certain to detach."""
    def make_ts() -> tuple[KTAM, State]: 
        ts = TileSet(
            [
                Tile([1, 1, 1, 1], name="tile1"),
                Tile([0, 2, 0, 2], name="tile2")
            ],
            glues=[(1,2,0.001)],
            canvas_type="square",
            size=(128, 128),
            alpha=-7.1,
            gse=10.1,
            gmc=20.0,
            fission="keep-weighted"
        )

        sys, state = cast(tuple[KTAM, State], ts.create_system_and_state())
        state.canvas_view[3:13, 3:13] = 1
        state.canvas_view[3:13, 14:34] = 1
        state.canvas_view[8, 13] = 2
        sys.update_all(state)
        return sys, state

    keep_left = 0
    keep_right = 0
    for i in range(500):
        sys, state = make_ts()
        sys.evolve(state, for_events=10, size_min=0, size_max=1000)
        if state.canvas_view[5, 20] == 1 and state.canvas_view[5, 10] == 0:
            keep_right += 1
        elif state.canvas_view[5, 20] == 0 and state.canvas_view[5, 10] == 1:
            keep_left += 1
        else:
            raise ValueError("No fission")
        
    assert keep_right > 1.5 * keep_left
    assert keep_right < 2.5 * keep_left

def test_ktam_fission_just_detach():
    """Test that JustDetach allows detachment without special fission handling."""
    ts = TileSet(
        [
            Tile([1, 1, 1, 1], name="tile1"),
            Tile([0, 2, 0, 2], name="tile2")
        ],
        glues=[(1,2,0.01)],
        canvas_type="square",
        size=(128, 128),
        alpha=-7.1,
        gse=20.1,
        gmc=40.0,
        fission="just-detach",
        seed=(4, 2, "tile1"),
    )

    sys, state = cast(tuple[KTAM, State], ts.create_system_and_state())
    state.canvas_view[3:13, 3:13] = 1
    state.canvas_view[3:13, 14:64] = 1
    state.canvas_view[8, 13] = 2
    sys.update_all(state)

    assert state.n_tiles > 600
    sys.evolve(state, for_events=2, size_min=0, size_max=1000)

    assert state.canvas_view[8, 13] == 0
    assert state.canvas_view[5, 20] == 1
    assert state.canvas_view[5, 10] == 1


def test_ktam_dimer_detach_off():
    """Test that JustDetach allows detachment without special fission handling."""
    ts = TileSet(
        [
            Tile([1, 1, 1, 1], name="tile1"),
            Tile([0, 2, "e3", 2], name="tile2"),
            Tile(["e3", 2, 0, 2], name="tile3"),
        ],
        bonds=[Bond("e3", 100 )],
        glues=[(1,2,0.01)],
        canvas_type="square",
        size=(128, 128),
        alpha=-7.1,
        gse=10.1,
        gmc=20.0,
        fission="keep-seeded",
        seed=(4, 2, "tile1"),
    )

    sys, state = cast(tuple[KTAM, State], ts.create_system_and_state())
    state.canvas_view[3:13, 3:13] = 1
    state.canvas_view[3:13, 14:64] = 1
    state.canvas_view[8, 13] = 2
    state.canvas_view[9, 13] = 3
    sys.update_all(state)

    assert state.n_tiles > 600
    sys.evolve(state, for_events=10, size_min=0, size_max=1000)

    assert state.canvas_view[8, 13] == 2
    assert state.canvas_view[9, 13] == 3
    assert state.canvas_view[5, 20] == 1
    assert state.canvas_view[5, 10] == 1

def test_ktam_dimer_detach_on():
    """Test that JustDetach allows detachment without special fission handling."""
    ts = TileSet(
        [
            Tile([1, 1, 1, 1], name="tile1"),
            Tile([0, 2, "e3", 2], name="tile2"),
            Tile(["e3", 2, 0, 2], name="tile3"),
        ],
        bonds=[Bond("e3", 100 )],
        glues=[(1,2,0.01)],
        canvas_type="square",
        size=(128, 128),
        alpha=-7.1,
        gse=10.1,
        gmc=20.0,
        fission="keep-seeded",
        chunk_handling="detach",
        chunk_size="dimer",
        seed=(4, 2, "tile1"),
    )

    sys, state = cast(tuple[KTAM, State], ts.create_system_and_state())
    state.canvas_view[3:13, 3:13] = 1
    state.canvas_view[3:13, 14:64] = 1
    state.canvas_view[8, 13] = 2
    state.canvas_view[9, 13] = 3
    sys.update_all(state)

    assert state.n_tiles > 600
    sys.evolve(state, for_events=10, size_min=0, size_max=1000)

    assert state.canvas_view[8, 13] == 0
    assert state.canvas_view[9, 13] == 0
    assert state.canvas_view[5, 20] == 0
    assert state.canvas_view[5, 10] == 1

