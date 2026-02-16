from rgrow.kblock import KBlock, KBlockTile, KBlockParams
from rgrow import EvolveOutcome, FFSRunConfig, State
import numpy as np

R_CONST = 1.98720425864083e-3  # kcal/mol/K
DS_LAT = -14.12 / 1000 # kcal/mol/K

def test_kblock_exactly_equal():
    blocker_conc = 0
    bind_strength = ((50+273.15) * R_CONST * np.log(1e-7)) / 2

    
    params = KBlockParams(
        [
            KBlockTile("ta", 1e-7, ["a","b*","b*","a"],),
            KBlockTile("tb", 1e-7, ["b","a*","a*","b"]),
        ],
        {"a": blocker_conc, "b": blocker_conc},
        seed={},
        binding_strength = {'a': bind_strength, 'b': bind_strength},
        # ds_lat = 0
    )
    sys = KBlock(params)

    state = State((10, 256), kind="tube")

    state.canvas_view[::2, 5:50] = 1 << 4
    state.canvas_view[1::2, 5:50] = 2 << 4

    sys.update_all(state)

    assert sys.calc_mismatches(state) == 0


def test_kblock_growth():
    blocker_conc = 1e-9
    bind_strength = ((50+273.15) * R_CONST * np.log(1e-7)) / 2 - 5

    
    params = KBlockParams(
        [
            KBlockTile("ta", 1e-7, ["a","b*","b*","a"],),
            KBlockTile("tb", 1e-7, ["b","a*","a*","b"]),
        ],
        {"a": blocker_conc, "b": blocker_conc},
        seed={},
        binding_strength = {'a': bind_strength, 'b': bind_strength},
        # ds_lat = 0
    )
    sys = KBlock(params)

    state = State((10, 256), kind="tube")

    state.canvas_view[::2, 5:50] = 1 << 4
    state.canvas_view[1::2, 5:50] = 2 << 4

    sys.update_all(state)

    assert sys.calc_mismatches(state) == 0

    out = sys.evolve(state, for_time=100000, size_max=2*state.n_tiles)

    assert out == EvolveOutcome.ReachedSizeMax

def test_kblock_melt():
    blocker_conc = 1e-9
    bind_strength = ((50+273.15) * R_CONST * np.log(1e-7)) / 2 + 4

    
    params = KBlockParams(
        [
            KBlockTile("ta", 1e-7, ["a","b*","b*","a"],),
            KBlockTile("tb", 1e-7, ["b","a*","a*","b"]),
        ],
        {"a": blocker_conc, "b": blocker_conc},
        seed={},
        binding_strength = {'a': bind_strength, 'b': bind_strength},
        # ds_lat = 0
    )
    sys = KBlock(params)

    state = State((10, 256), kind="tube")

    state.canvas_view[::2, 5:50] = 1 << 4
    state.canvas_view[1::2, 5:50] = 2 << 4

    sys.update_all(state)

    assert sys.calc_mismatches(state) == 0

    out = sys.evolve(state, for_time=100000, size_max=2*state.n_tiles, size_min=state.n_tiles)

    assert out == EvolveOutcome.ReachedSizeMin


def test_kblock_frozen():
    blocker_conc = 1e-5
    bind_strength = ((50+273.15) * R_CONST * np.log(1e-7)) / 2 - 5

    
    params = KBlockParams(
        [
            KBlockTile("ta", 1e-7, ["a","b*","b*","a"],),
            KBlockTile("tb", 1e-7, ["b","a*","a*","b"]),
        ],
        {"a": blocker_conc, "b": blocker_conc},
        seed={},
        binding_strength = {'a': bind_strength, 'b': bind_strength},
        # ds_lat = 0
    )
    sys = KBlock(params)

    state = State((10, 256), kind="tube")

    state.canvas_view[::2, 5:50] = 1 << 4
    state.canvas_view[1::2, 5:50] = 2 << 4

    sys.update_all(state)

    assert sys.calc_mismatches(state) == 0

    out = sys.evolve(state, for_time=100000, size_max=int(1.1*state.n_tiles))

    assert out == EvolveOutcome.ReachedTimeMax

def test_kblock_high_blocker_no_nucleation():
    """In favorable growth conditions, with high blockers, the system should not nucleate."""
    blocker_conc = 2e-6
    bind_strength = ((50+273.15) * R_CONST * np.log(1e-7)) / 2 - 4

    
    params = KBlockParams(
        [
            KBlockTile("ta", 1e-7, ["a","b*","b*","a"],),
            KBlockTile("tb", 1e-7, ["b","a*","a*","b"]),
        ],
        {"a": blocker_conc, "b": blocker_conc},
        seed={},
        binding_strength = {'a': bind_strength, 'b': bind_strength},
        # ds_lat = 0
    )
    sys = KBlock(params)

    ffs_config = FFSRunConfig(
        min_nuc_rate=1e-12
    )

    res =  sys.run_ffs(ffs_config)

    assert res.nucleation_rate < 1e-11

def test_kblock_low_blocker_sp_nucleation():
    """In favorable growth conditions, with no blockers, the system should nucleate."""
    blocker_conc = 0
    bind_strength = ((50+273.15) * R_CONST * np.log(1e-7)) / 2 - 4

    
    params = KBlockParams(
        [
            KBlockTile("ta", 1e-7, ["a","b*","b*","a"],),
            KBlockTile("tb", 1e-7, ["b","a*","a*","b"]),
        ],
        {"a": blocker_conc, "b": blocker_conc},
        seed={},
        binding_strength = {'a': bind_strength, 'b': bind_strength},
        # ds_lat = 0
    )
    sys = KBlock(params)

    ffs_config = FFSRunConfig(
        min_nuc_rate=1e-12
    )

    res =  sys.run_ffs(ffs_config)

    assert res.nucleation_rate > 1e-9