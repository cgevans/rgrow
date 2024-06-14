from rgrow import Tile, TileSet, Bond  # noqa: F841
import pytest # noqa: F401
from pytest import approx
import hypothesis.strategies as st
from hypothesis import given
import numpy as np


@given(
    gse=st.floats(4, 12),
    concs_nM=st.floats(1, 10000),
    alpha=st.floats(-10, 10),
    kf=st.floats(1e3, 1e9),
    stoic=st.floats(0.1, 5),
    bond_strength=st.floats(0.5, 3)
)
def test_basic_rates_ktam(gse, concs_nM, alpha, kf, stoic, bond_strength):
    gmc = alpha-np.log(concs_nM / 1e9)

    ts = TileSet([
        Tile(["a", "b", "c", "d"], stoic = stoic),
        Tile(["e", "d", "g", "b"]),
        ],
        [Bond("b", bond_strength)],
        [("e", "g", 1)],
        kf=kf,
        alpha=alpha,
        gse=gse,
        gmc=gmc,
        )
    
    sys, state = ts.create_system_and_state()

    cv = state.canvas_view

    cv[5,5] = 1
    cv[5,6] = 2
    cv[6,6] = 2

    sys.update_all(state)

    dimers = sys.calc_dimers()

    for d in dimers:
        if (d.t1 == 1) and (d.t2 == 1):
            assert d.formation_rate == approx(kf * (concs_nM / 1e9)**2 * stoic**2)
        elif (d.t1 == 1) and (d.t2 == 2):
            assert d.formation_rate == approx(kf * (concs_nM / 1e9)**2 * stoic)
        elif (d.t1 == 2) and (d.t2 == 1):
            assert d.formation_rate == approx(kf * (concs_nM / 1e9)**2 * stoic)
        elif (d.t1 == 2) and (d.t2 == 2):
            assert d.formation_rate == approx(kf * (concs_nM / 1e9)**2)
        else:
            raise ValueError("Unexpected Dimer: ", d)

    assert state.rate_at_point((5,5)) == approx(kf * np.exp(-bond_strength * gse + alpha))

    assert state.rate_at_point((5,6)) == approx(kf * np.exp(-(1+bond_strength) * gse + alpha))

    assert state.rate_at_point((6,6)) == approx(kf * np.exp( -gse + alpha))

    assert state.rate_at_point((6,5)) == approx(kf * stoic * concs_nM / 1e9)


@given(
    gse=st.floats(4, 12),
    concs_nM=st.floats(1, 10000),
    alpha=st.floats(-10, 10),
    kf=st.floats(1e3, 1e9),
    stoic=st.floats(0.1, 5),
    bond_strength=st.floats(0.5, 3)
)
def test_basic_rates_oldktam(gse, concs_nM, alpha, kf, stoic, bond_strength):
    gmc = alpha-np.log(concs_nM / 1e9)

    ts = TileSet([
        Tile(["a", "b", "c", "d"], stoic = stoic),
        Tile(["e", "d", "g", "b"]),
        ],
        [Bond("b", bond_strength)],
        [("e", "g", 1)],
        kf=kf,
        alpha=alpha,
        gse=gse,
        gmc=gmc,
        model='oldktam'
        )
    
    sys, state = ts.create_system_and_state()

    cv = state.canvas_view

    cv[5,5] = 1
    cv[5,6] = 2
    cv[6,6] = 2

    sys.update_all(state)

    dimers = sys.calc_dimers()

    for d in dimers:
        if (d.t1 == 1) and (d.t2 == 1):
            assert d.formation_rate == approx(kf * (concs_nM / 1e9)**2 * stoic**2)
        elif (d.t1 == 1) and (d.t2 == 2):
            assert d.formation_rate == approx(kf * (concs_nM / 1e9)**2 * stoic)
        elif (d.t1 == 2) and (d.t2 == 1):
            assert d.formation_rate == approx(kf * (concs_nM / 1e9)**2 * stoic)
        elif (d.t1 == 2) and (d.t2 == 2):
            assert d.formation_rate == approx(kf * (concs_nM / 1e9)**2)
        else:
            raise ValueError("Unexpected Dimer: ", d)

    assert state.rate_at_point((5,5)) == approx(kf * np.exp(-bond_strength * gse + alpha))

    assert state.rate_at_point((5,6)) == approx(kf * np.exp(-(1+bond_strength) * gse + alpha))

    assert state.rate_at_point((6,6)) == approx(kf * np.exp( -gse + alpha))

    assert state.rate_at_point((6,5)) == approx(kf * stoic * concs_nM / 1e9)
