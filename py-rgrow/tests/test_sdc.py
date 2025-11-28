from rgrow import State
from rgrow.sdc import SDCParams, SDCStrand, SDC
import numpy as np

def mean_energies(strand_length: int = 10) -> tuple[float, float]:
    return (
        -1.405625 * (strand_length-1),
        -0.02201875 * (strand_length-1)
    )

def make_bitcopy(N, input="0", conc=1e-7, cdl=10, sdl=20):
    strands = []

    match input:
        case "0":
            conc0 = 1e-7
            conc1 = 0
        case "1":
            conc0 = 0
            conc1 = 1e-7


    strands.append(SDCStrand(conc0, "c0", "sc0", "c0*", "input0"))
    strands.append(SDCStrand(conc1, "c1", "sc0", "c1*", "input1"))

    for i in range(1,N):
        strands.append(SDCStrand(conc, "c0", f"sc{i}", "c0*", f"{i}_0"))
        strands.append(SDCStrand(conc, "c1", f"sc{i}", "c1*", f"{i}_1"))

    params = SDCParams(
        strands=strands,
        glue_dg_s = (
            {"c0": mean_energies(cdl)} |
            {"c1": mean_energies(cdl)} |
            {f"sc{i}": mean_energies(sdl) for i in range(0,N)}
        ),
        scaffold = [f"sc{i}*" for i in range(0,N)]
    )

    return SDC(params)

def temperature_for_target_prob(sys, prob=0.9, precision=0.1, start_range=(60, 90)):
    sys.temperature = 0
    target = sys.mfe_config()[0]
    low, high = start_range
    while high - low > precision:
        mid = (low + high) / 2
        sys.temperature = mid
        p = sys.probability_of_state(target)
        if p > prob:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def test_bitcopy_const_temp():
    N = 8
    sys = make_bitcopy(N)
    state = State((1024, N+4), kind="Square", tracking="None", n_tile_types=len(sys.tile_names))
    sys.update_state(state)

    sys.temperature = temperature_for_target_prob(sys, prob=0.8, precision=0.1)

    sys.evolve(state, for_events=10000000)

    target = np.all(state.canvas_view[2:-2,:] == np.array(sys.mfe_config()[0]), axis=1).mean()
    assert target > 0.70
    assert target < 0.90
   
    # Now test melting:
    sys.temperature += 15
    sys.evolve(state, for_events=10000000)

    assert state.n_tiles < 10

def test_bitcopy_sane_temps():
    N = 8
    sys = make_bitcopy(N)

    # At 30, MFE config should be the target state.
    sys.temperature = 30
    target_config = sys.mfe_config()[0]

    known_zero_target = [0,0,1] + [2*i+1 for i in range(1, N)] + [0,0]
    # known_one_target = [0,0,2] + [2*i+2 for i in range(1, N)] + [0,0]
    assert target_config == known_zero_target

    # Our probability should be high:
    prob = sys.probability_of_state(target_config)
    assert prob > 0.95

    sys.temperature = 90

    prob = sys.probability_of_state(target_config)
    assert prob < 0.05

    # Our MFE should be empty:
    mfe_config = sys.mfe_config()[0]
    assert np.all(mfe_config == [0,0] + [0]*N + [0,0])

    # As we decrease temperature, the probability of the target config should monotonically increase
    for temp in range(85, 20, -10):
        sys.temperature = temp
        new_prob = sys.probability_of_state(target_config)
        assert new_prob > prob
        assert np.allclose(np.exp(-sys.state_g(target_config) / sys.rtval())/sys.partition_function(), new_prob)
        prob = new_prob

    sys.temperature = 30
    pf = sys.partition_function()
    _, en = sys.mfe_config()
    assert np.allclose(np.exp(-en / sys.rtval()), pf)


def test_basic_on_rates():
    N = 8
    sys = make_bitcopy(N)
    state = State((1024, N+4), kind="Square", tracking="None", n_tile_types=len(sys.tile_names))
    sys.update_state(state)

    kf = sys.get_param("kf")

    assert state.rate_at_point((5, 2)) == kf * 1e-7
    assert state.rate_at_point((5, 3)) == 2 * kf * 1e-7

    sys.set_param("kf", 1e7)
    sys.update_state(state)
    assert state.rate_at_point((5, 2)) == 1e7 * 1e-7
    assert state.rate_at_point((5, 3)) == 2 * 1e7 * 1e-7
