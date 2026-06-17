from typing import cast
from rgrow import EvolveOutcome, Tile, TileSet, Bond  # noqa: F841
import pytest  # noqa: F401
from rgrow import KTAM, State

def test_ktam_growth():
    """A simple A/B checkerboard tile system, on a tube.  This should grow if two bonds are favorable, should shrink if two bonds are unfavorable,
    and should be at equilibrium at gmc=2*gse."""
    import polars as pl
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
        gmc=10.0,
        tracking={"type": "energychanges", "bin_width": 1e-12}
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

    t = cast(pl.DataFrame, state.tracking_copy())

    # These comparison tolerances are quite important: default rel_tol is polars is 1e-9.
    two_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - 2*tube_ts.gse, abs_tol=1e-13))[0,'count']
    two_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - 2*tube_ts.gse), abs_tol=1e-13))[0,'count']
    one_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - tube_ts.gse, abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    one_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - tube_ts.gse), abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    three_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - 3*tube_ts.gse, abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    three_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - 3*tube_ts.gse), abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    four_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - 4*tube_ts.gse, abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    four_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - 4*tube_ts.gse), abs_tol=1e-15, rel_tol=1e-16))[0,'count']

    end_n = state.n_tiles
    tiles_gained = end_n - start_n

    # Most growth should come from net two-bond attachments
    two_bond_net = two_bond_att - two_bond_det
    assert two_bond_net > 0, f"Expected net two-bond growth, got net {two_bond_net}"
    assert two_bond_net >= 0.8 * tiles_gained, (
        f"Two-bond net ({two_bond_net}) should account for most growth ({tiles_gained})"
    )

    # One-bond attach and detach should be approximately equal
    assert abs(one_bond_att - one_bond_det) < 0.1 * (one_bond_att + one_bond_det), (
        f"One-bond should be roughly balanced: att={one_bond_att}, det={one_bond_det}"
    )

    # Three and four bond events should be rare compared to two-bond
    assert three_bond_att + three_bond_det < 0.1 * (two_bond_att + two_bond_det)
    assert four_bond_att + four_bond_det < 0.1 * (two_bond_att + two_bond_det)

def test_ktam_melt():
    import polars as pl
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
        gmc=10.0,
        tracking={"type": "energychanges", "bin_width": 1e-15}
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
    out = sys.evolve(state, for_events=100_000, size_min=1, size_max=2*start_n)

    t = cast(pl.DataFrame, state.tracking_copy())

    two_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - 2*tube_ts.gse))[0,'count']
    two_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - 2*tube_ts.gse)))[0,'count']
    one_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - tube_ts.gse))[0,'count']
    one_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - tube_ts.gse)))[0,'count']
    three_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - 3*tube_ts.gse))[0,'count']
    three_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - 3*tube_ts.gse)))[0,'count']
    four_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - 4*tube_ts.gse))[0,'count']
    four_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - 4*tube_ts.gse)))[0,'count']

    end_n = state.n_tiles
    tiles_lost = start_n - end_n

    # We should have reached the min size:
    assert out == EvolveOutcome.ReachedSizeMin

    end_n = state.n_tiles
    tiles_lost = start_n - end_n

    # Most melting should come from net two-bond detachments
    two_bond_net = two_bond_att - two_bond_det
    assert two_bond_net < 0, f"Expected net two-bond melting, got net {two_bond_net}"
    assert abs(two_bond_net) >= 0.8 * tiles_lost, (
        f"Two-bond net ({two_bond_net}) should account for most melting ({tiles_lost})"
    )

    # One-bond attach and detach should be approximately equal
    # (tiles attach at frontier and quickly fall off)
    assert abs(one_bond_att - one_bond_det) < 0.1 * (one_bond_att + one_bond_det), (
        f"One-bond should be roughly balanced: att={one_bond_att}, det={one_bond_det}"
    )

    # Three and four bond events should be rare compared to two-bond
    assert three_bond_att + three_bond_det < 0.1 * (two_bond_att + two_bond_det)
    assert four_bond_att + four_bond_det < 0.1 * (two_bond_att + two_bond_det)

def test_ktam_equilibrium():
    """A perfectly balanced system (gmc=2*gse) should remain stable: neither
    growing to max size nor melting completely.  We use a large starting
    structure so the boundaries are far from the random walk's reach
    (stddev ≈ sqrt(100k) ≈ 316, boundaries ~1560 tiles away)."""

    import polars as pl
    tube_ts = TileSet(
        [
            Tile(["a","a","b","b"],),
            Tile(["b","b","a","a"]),
        ],
        [Bond("a", 1), Bond("b", 1)],
        canvas_type="tube",
        size=(8, 512),
        alpha=-7.1,
        gse=5.0 + 5e-11,  # We set an extremely small bias to make energy change show two bond att/det
        gmc=10.0,
        tracking={"type": "energychanges", "bin_width": 1e-12},
        fission="no-fission"
    )

    sys, state = cast(tuple[KTAM, State], tube_ts.create_system_and_state())

    state.canvas_view[::2, 5:200] = 1
    state.canvas_view[1::2, 5:200] = 2

    sys.update_all(state)

    assert sys.calc_mismatches(state) == 0

    start_n = state.n_tiles

    out = sys.evolve(state, for_events=100_000, size_min=0, size_max=2*start_n)
    
    t = cast(pl.DataFrame, state.tracking_copy())


    two_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - 2*tube_ts.gse, abs_tol=1e-13))[0,'count']
    two_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - 2*tube_ts.gse), abs_tol=1e-13))[0,'count']
    one_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - tube_ts.gse, abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    one_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - tube_ts.gse), abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    three_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - 3*tube_ts.gse, abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    three_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - 3*tube_ts.gse), abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    four_bond_att = t.filter(pl.col("energy_change").is_close(tube_ts.gmc - 4*tube_ts.gse, abs_tol=1e-15, rel_tol=1e-16))[0,'count']
    four_bond_det = t.filter(pl.col("energy_change").is_close(-(tube_ts.gmc - 4*tube_ts.gse), abs_tol=1e-15, rel_tol=1e-16))[0,'count']

    assert out == EvolveOutcome.ReachedEventsMax

    # At equilibrium, two-bond attach and detach should be roughly equal
    assert two_bond_att > 0 and two_bond_det > 0
    two_bond_ratio = two_bond_att / two_bond_det
    assert 0.96 < two_bond_ratio < 1.04, (
        f"Two-bond attach/detach should be ~equal at equilibrium: "
        f"{two_bond_att}/{two_bond_det} = {two_bond_ratio:.4f}"
    )

    changerate = abs(two_bond_att-two_bond_det) / (two_bond_att + two_bond_det)

    assert changerate < 0.01, (
        f"Two-bond attach and detach should be within 1% of total events: att={two_bond_att}, det={two_bond_det}, {changerate:.4%} difference"
    )

    # Three and four bond events should be rare compared to two-bond
    assert three_bond_att + three_bond_det < 0.1 * (two_bond_att + two_bond_det)
    assert four_bond_att + four_bond_det < 0.1 * (two_bond_att + two_bond_det)

    # One-bond should be frequent and roughtly balanced (tiles attach at frontier and quickly fall off)
    assert one_bond_att > 0 and one_bond_det > 0
    one_bond_ratio = one_bond_att / one_bond_det
    assert 0.96 < one_bond_ratio < 1.04, (
        f"One-bond attach/detach should be ~equal at equilibrium: "
        f"{one_bond_att}/{one_bond_det} = {one_bond_ratio:.4f}"
    )

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



# --- ChunkHandling coexistence on a tube of two-tile blocks --------------------
#
# A/B blocks tile a cylinder: each block is an A-B pair bound internally by a
# double-strength bond ("d", strength 2); blocks bind their neighbours with
# single bonds ("s", "w").  One vertical bond per tile is null, so each tile's
# bond-strength sum is 4 (energy-per-tile = 2*gse), making gmc = 2*gse the
# coexistence point for *single-tile* dynamics.  The tile design is derived from
# the existing mismatch-free tube checkerboard (Tile["a","a","b","b"] /
# Tile["b","b","a","a"]) by splitting its glues and reassigning strengths.
#
# (calc_mismatches is intentionally not asserted to be 0: the null bonds are
# non-bonding adjacencies, which calc_mismatches counts, but they are not glue
# conflicts.)

_CHUNK_GSE = 5.0


def _chunk_block_tileset(chunk_handling, chunk_size, *, gmc, alpha=-7.1):
    return TileSet(
        [
            Tile([0, "d", "s", "w"], name="A"),  # N=null, E=d(2), S=s(1), W=w(1)
            Tile(["s", "w", 0, "d"], name="B"),  # N=s(1), E=w(1), S=null, W=d(2)
        ],
        [Bond("d", 2), Bond("s", 1), Bond("w", 1)],
        canvas_type="tube",
        size=(8, 512),
        alpha=alpha,
        gse=_CHUNK_GSE,
        gmc=gmc,
        chunk_handling=chunk_handling,
        chunk_size=chunk_size,
        fission="no-fission",
    )


def _run_chunk_mode(chunk_handling, chunk_size, gmc, events=100_000):
    ts = _chunk_block_tileset(chunk_handling, chunk_size, gmc=gmc)
    sys, state = cast(tuple[KTAM, State], ts.create_system_and_state())
    state.canvas_view[::2, 5:250] = 1
    state.canvas_view[1::2, 5:250] = 2
    sys.update_all(state)
    start_n = state.n_tiles
    outcome = sys.evolve(state, for_events=events, size_min=1, size_max=3 * start_n)
    return {"start": start_n, "end": state.n_tiles, "outcome": outcome}


def test_ktam_chunk_detailed_balance():
    """Three-way coexistence test for ChunkHandling, each mode at its own balance.

    - None at gmc = 2*gse (its coexistence: energy-per-tile = 2*gse) -> the
      structure random-walks.
    - Detach at gmc = 2*gse -> chunk (dimer) detachment is added with no matching
      chunk attachment, so detailed balance is broken and the structure melts.
    - Equilibrium at gmc = 2*gse - ln(2) -> chunk attachment is added back from the
      equilibrium dimer concentrations, restoring detailed balance.  Its coexistence
      is shifted DOWN from 2*gse by ln(2): the dominant double-bond dimer has
      E_internal = 2*gse equal to the two single bonds a front monomer makes, so at
      balance exactly half the monomers are tied up in dimers (total = 2*free) ->
      gmc = 2*gse - ln(2).  This shift is alpha-independent.  There the structure
      random-walks; at gmc = 2*gse it would melt (see the note below).

    The chunk attach/detach *rate* balance itself is checked, alpha-independently,
    by test_ktam_we_dimer_detach_rates and test_ktam_equilibrium_attachment_conservation.
    """
    import math

    GSE = _CHUNK_GSE
    none = _run_chunk_mode("none", "single", gmc=2 * GSE)
    detach = _run_chunk_mode("detach", "dimer", gmc=2 * GSE)
    equil = _run_chunk_mode("equilibrium", "dimer", gmc=2 * GSE - math.log(2))

    # None random-walks at its coexistence (does not melt to the floor or grow to
    # the cap, and stays near its starting size).
    assert none["outcome"] == EvolveOutcome.ReachedEventsMax, none
    assert none["start"] * 0.5 < none["end"] < none["start"] * 1.6, none

    # Detach melts: the unmatched chunk detachment drives the structure down.
    assert detach["end"] < detach["start"] * 0.5, detach

    # Equilibrium random-walks at its (depletion-shifted) coexistence.
    assert equil["outcome"] == EvolveOutcome.ReachedEventsMax, equil
    assert equil["start"] * 0.5 < equil["end"] < equil["start"] * 1.6, equil


def _equilibrium_2bond_attach_free_energy(gmc, alpha=-7.1):
    """Corrected (free-concentration) energy change of a 2-bond monomer attach.

    With the energy fix, the chemical-potential term uses the depleted free
    concentration, so this equals gmc_eff - 2*gse where gmc_eff = -ln(free) + alpha.
    free is the free monomer concentration from the equilibrium dimer solve.
    Zero at coexistence.
    """
    import math

    ts = _chunk_block_tileset("equilibrium", "dimer", gmc=gmc, alpha=alpha)
    sys, _ = cast(tuple[KTAM, State], ts.create_system_and_state())
    total = math.exp(-gmc + alpha)  # stoic 1
    in_dimers = sum(float(d.equilibrium_conc) for d in sys.calc_dimers() if d.t1 == 1 or d.t2 == 1)
    free = total - in_dimers
    gmc_eff = -math.log(free) + alpha
    return gmc_eff - 2 * _CHUNK_GSE


def test_ktam_equilibrium_coexistence_energy():
    """The energy fix reveals Equilibrium's coexistence point.

    The corrected 2-bond-attach free energy is ~+ln(golden ratio) (positive ->
    melt) at gmc = 2*gse, and ~0 (balanced) at gmc = 2*gse - ln(2).  This is what
    makes the gmc shift in test_ktam_chunk_detailed_balance visible/quantitative,
    and it is alpha-independent.
    """
    import math

    GSE = _CHUNK_GSE
    phi = (1 + 5**0.5) / 2  # at gmc=2*gse, free/total = 1/phi (golden ratio)
    for alpha in (-7.1, 0.0, 2.0):
        e_2gse = _equilibrium_2bond_attach_free_energy(2 * GSE, alpha=alpha)
        e_bal = _equilibrium_2bond_attach_free_energy(2 * GSE - math.log(2), alpha=alpha)
        assert abs(e_2gse - math.log(phi)) < 0.03, (alpha, e_2gse, math.log(phi))
        assert abs(e_bal) < 0.03, (alpha, e_bal)


def test_ktam_equilibrium_attachment_conservation():
    """Total attachment rate is conserved between None and Equilibrium, less blocked dimers.

    A attaches east of a placed seed S (A.W binds S.E); A's only dimer is the WE
    dimer (A,B) extending further east (A.E binds B.W). S has concentration 0, so
    it forms no dimers and A is depleted only by (A,B).

    - When the (A,B) dimer fits (A's east neighbour empty), Equilibrium's total
      attachment rate at the site equals None's: A's depletion is exactly offset
      by the dimer's attachment.
    - When the dimer is blocked (east neighbour occupied), Equilibrium falls short
      of None by exactly the blocked dimer's rate, kf*[dimer].

    Checked across several alphas (the depletion bookkeeping must be
    alpha-independent).
    """
    GSE, GMC, KF = 3.0, 6.9, 1e6  # conc ~ exp(-gmc+alpha)

    def build(chunk_handling, alpha):
        return TileSet(
            [
                Tile([0, "d", 0, "x"], name="A"),  # N=0, E=d(2), S=0, W=x(1)
                Tile([0, 0, 0, "d"], name="B"),    # W=d(2)
                Tile([0, "x", 0, 0], name="S", stoic=0.0),  # E=x(1); conc 0 -> no dimers
            ],
            [Bond("d", 2), Bond("x", 1)],
            canvas_type="square",
            size=(16, 16),
            gse=GSE,
            gmc=GMC,
            alpha=alpha,
            kf=KF,
            chunk_handling=chunk_handling,
            chunk_size="dimer",
        )

    def total_at(chunk_handling, alpha, blocker):
        ts = build(chunk_handling, alpha)
        sys, state = cast(tuple[KTAM, State], ts.create_system_and_state())
        state.canvas_view[8, 8] = 3  # seed S, so A can attach at (8, 9)
        if blocker:
            state.canvas_view[8, 10] = 2  # occupy A's east neighbour -> (A,B) blocked
        sys.update_all(state)
        return sys, state.rate_at_point((8, 9))

    for alpha in (0.0, -3.0, 2.0):
        _, none_rate = total_at("none", alpha, blocker=False)
        eq_sys, eq_fits = total_at("equilibrium", alpha, blocker=False)
        _, eq_blocked = total_at("equilibrium", alpha, blocker=True)
        _, none_blocked = total_at("none", alpha, blocker=True)

        dimers = eq_sys.calc_dimers()
        dimer_conc = next(float(d.equilibrium_conc) for d in dimers if d.t1 == 1 and d.t2 == 2)
        kf_dimer = KF * dimer_conc
        assert dimer_conc > 0, f"alpha={alpha}: expected nonzero depletion"

        # Dimer fits: A's depletion is exactly offset by the dimer's attachment.
        assert eq_fits == pytest.approx(none_rate, rel=1e-7), (alpha, eq_fits, none_rate)
        # None's rate at the site is unaffected by the (downstream) blocker.
        assert none_blocked == pytest.approx(none_rate, rel=1e-9), (alpha, none_blocked, none_rate)
        # Dimer blocked: Equilibrium is short of None by exactly the blocked dimer's rate.
        assert eq_blocked < none_blocked
        assert none_blocked - eq_blocked == pytest.approx(kf_dimer, rel=1e-6), (
            alpha,
            none_blocked,
            eq_blocked,
            kf_dimer,
        )
