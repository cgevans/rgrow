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

    strands.append(SDCStrand(conc, f"c{input}", "sc0", f"c{input}*", f"input{input}"))

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

def test_basic_on_rates():
    N = 8
    sys = make_bitcopy(N)
    state = State((1024, N+4), kind="Square", tracking="None", n_tile_types=len(sys.tile_names))
    sys.update_state(state)

    # TODO: get kf directly
    kf = 1e6

    assert state.rate_at_point((5, 2)) == kf * 1e-7
    assert state.rate_at_point((5, 3)) == 2 * kf * 1e-7