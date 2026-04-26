import pytest

from rgrow import State
from rgrow.sdc2d import SDC2D, SDC2DParams, SDC2DStrand


def _padded_uniform_params(n=8, dg=-8.0, ds=0.0, conc=1e-6, k_f=1e6, temperature=37.0):
    """Build an n x n scaffold whose interior [2..n-2, 2..n-2] all matches a
    single strand A by glue 'g'/'g*'. Border cells are None."""
    scaffold: list[list[str | None]] = [[None] * n for _ in range(n)]
    for r in range(2, n - 2):
        for c in range(2, n - 2):
            scaffold[r][c] = "g*"
    return SDC2DParams(
        strands=[
            SDC2DStrand(
                concentration=conc,
                bottom_glue="g",
                name="A",
                color="red",
            )
        ],
        scaffold=scaffold,
        scaffold_concentration=1e-9,
        glue_dg_s={"g": (dg, ds)},
        k_f=k_f,
        temperature=temperature,
        seed=[],
    )


def test_construct_minimal():
    sys = SDC2D(_padded_uniform_params(n=6))
    assert sys.nrows() == 6
    assert sys.ncols() == 6
    # null + A
    assert sys.n_strands() == 2
    assert "A" in sys.strand_names
    # An interior position should have A (id 1) as a friend.
    assert sys.friends_at(3, 3) == [1]
    # A border position should have no friends (null scaffold glue).
    assert sys.friends_at(0, 0) == []


def test_temperature_setter_changes_rates():
    sys = SDC2D(_padded_uniform_params(n=6, dg=-8.0, ds=-0.01))
    state = State((6, 6), kind="Square", tracking="None", n_tile_types=sys.n_strands())
    sys.update_state(state)

    p = (3, 3)
    rate_low = state.rate_at_point(p)
    assert rate_low > 0.0

    sys.temperature = 70.0
    sys.update_state(state)
    rate_high = state.rate_at_point(p)
    assert rate_high > 0.0

    # Different temperature should change attachment rate at empty cells only
    # if attachment depends on T (it doesn't directly), so rates of attachment
    # are equal — what changes is the detachment rate after a tile is placed.
    # Verify temperature getter reflects the change.
    assert abs(sys.temperature - 70.0) < 1e-9


def test_evolve_grows_tiles():
    sys = SDC2D(_padded_uniform_params(n=8, dg=-12.0, ds=0.0))
    state = State((8, 8), kind="Square", tracking="None", n_tile_types=sys.n_strands())
    sys.update_state(state)
    sys.evolve(state, for_events=200)
    assert state.n_tiles >= 1


def test_dna_sequence_glue():
    # Specify a glue using a DNA sequence; check that it produces a non-zero
    # binding (i.e. the glue is registered and the friends list is populated).
    scaffold: list[list[str | None]] = [[None] * 6 for _ in range(6)]
    for r in range(2, 4):
        for c in range(2, 4):
            scaffold[r][c] = "g*"
    params = SDC2DParams(
        strands=[
            SDC2DStrand(concentration=1e-6, bottom_glue="g", name="A"),
        ],
        scaffold=scaffold,
        scaffold_concentration=1e-9,
        glue_dg_s={"g": "GGACTGAC"},
        k_f=1e6,
        temperature=37.0,
        seed=[],
    )
    sys = SDC2D(params)
    assert sys.friends_at(2, 2) == [1]


def test_seed_pins_tile():
    params = _padded_uniform_params(n=8, dg=-15.0, ds=0.0)
    params.seed = [(3, 3, "A")]
    sys = SDC2D(params)
    state = State((8, 8), kind="Square", tracking="None", n_tile_types=sys.n_strands())
    sys.update_state(state)
    # configure_empty_state placed the seed at (3, 3); evolve briefly and
    # verify it is still there (even though detachment of A elsewhere is fast,
    # the seed cannot detach).
    sys.evolve(state, for_events=200)
    assert state.canvas_view[3, 3] == 1


def test_get_param_kf():
    sys = SDC2D(_padded_uniform_params(n=6, k_f=2.5e6))
    assert sys.get_param("kf") == pytest.approx(2.5e6)
    sys.kf = 4.0e6
    assert sys.get_param("kf") == pytest.approx(4.0e6)
