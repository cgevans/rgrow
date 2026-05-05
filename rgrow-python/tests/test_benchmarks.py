import numpy as np
from rgrow import State
from rgrow.kblock import KBlock, KBlockTile, KBlockParams
from test_sdc import make_bitcopy, temperature_for_target_prob
from test_sierpinski import create_sierpinski_tileset

R_CONST = 1.98720425864083e-3


def test_bench_evolve_squarecompact(benchmark):
    """Benchmark evolve SDC N=8 bitcopy with SquareCompact canvas."""
    N = 8
    sys = make_bitcopy(N)
    temp = temperature_for_target_prob(sys, prob=0.75, precision=0.1)
    sys.temperature = temp

    def setup():
        state = State(
            (1024, N), kind="SquareCompact", tracking="None",
            n_tile_types=len(sys.tile_names),
        )
        sys.update_state(state)
        return (state,), {}

    def run(state):
        sys.evolve(state, for_events=1000000)

    benchmark.pedantic(run, setup=setup, rounds=5, warmup_rounds=1)


def test_bench_evolve_square(benchmark):
    """Benchmark evolve SDC N=8 bitcopy with Square canvas."""
    N = 8
    sys = make_bitcopy(N, pad=True)
    temp = temperature_for_target_prob(sys, prob=0.75, precision=0.1)
    sys.temperature = temp

    def setup():
        state = State(
            (1024, N + 4), kind="Square", tracking="None",
            n_tile_types=len(sys.tile_names),
        )
        sys.update_state(state)
        return (state,), {}

    def run(state):
        sys.evolve(state, for_events=1000000)

    benchmark.pedantic(run, setup=setup, rounds=5, warmup_rounds=1)


def test_bench_evolve_sierpinski_square(benchmark):
    """Benchmark evolve KTAM sierpinski with Square canvas (1020x1020, 50K events).

    Uses `setup_state` to place the seed; `update_state` alone leaves the
    canvas empty and evolve hits ReachedZeroRate immediately.
    """
    sys = create_sierpinski_tileset(pad_seed=True).create_system()

    def setup():
        state = State(
            (1020, 1020), kind="Square", tracking="None",
            n_tile_types=len(sys.tile_names),
        )
        sys.setup_state(state)
        return (state,), {}

    def run(state):
        sys.evolve(state, for_events=50000)

    benchmark.pedantic(run, setup=setup, rounds=7, warmup_rounds=1)


def test_bench_evolve_sierpinski_squarecompact(benchmark):
    """Benchmark evolve KTAM sierpinski with SquareCompact canvas (1020x1020, 50K events)."""
    sys = create_sierpinski_tileset(pad_seed=False).create_system()

    def setup():
        state = State(
            (1020, 1020), kind="SquareCompact", tracking="None",
            n_tile_types=len(sys.tile_names),
        )
        sys.setup_state(state)
        return (state,), {}

    def run(state):
        sys.evolve(state, for_events=50000)

    benchmark.pedantic(run, setup=setup, rounds=7, warmup_rounds=1)


def test_bench_evolve_sierpinski_periodic(benchmark):
    """Benchmark evolve KTAM sierpinski with Periodic canvas (1024x1024, 50K events).

    Exercises a third StateEnum dispatch arm (CanvasPeriodic × NullStateTracker)
    relative to Square / SquareCompact. Useful for catching changes that affect
    dispatch overhead asymmetrically across canvas variants (P10 / P11).
    """
    sys = create_sierpinski_tileset(pad_seed=False).create_system()

    def setup():
        state = State(
            (1024, 1024), kind="Periodic", tracking="None",
            n_tile_types=len(sys.tile_names),
        )
        sys.setup_state(state)
        return (state,), {}

    def run(state):
        sys.evolve(state, for_events=50000)

    benchmark.pedantic(run, setup=setup, rounds=7, warmup_rounds=1)


def test_bench_evolve_kblock_tube(benchmark):
    """Benchmark evolve KBlock 10x256 tube with seeded growth (200K events).

    The plan flagged StateEnum::tile_to_n at ~3% in this exact configuration
    (KBlock-tube, cdylib build) due to the 30-variant State dispatch table not
    inlining. This bench is the primary regression detector for P10 / P11.
    """
    blocker_conc = 1e-9
    bind_strength = ((50 + 273.15) * R_CONST * np.log(1e-7)) / 2 - 5
    params = KBlockParams(
        [
            KBlockTile("ta", 1e-7, ["a", "b*", "b*", "a"]),
            KBlockTile("tb", 1e-7, ["b", "a*", "a*", "b"]),
        ],
        {"a": blocker_conc, "b": blocker_conc},
        seed={},
        binding_strength={"a": bind_strength, "b": bind_strength},
    )
    sys = KBlock(params)

    def setup():
        state = State((10, 256), kind="tube")
        state.canvas_view[::2, 5:50] = 1 << 4
        state.canvas_view[1::2, 5:50] = 2 << 4
        sys.update_all(state)
        return (state,), {}

    def run(state):
        sys.evolve(state, for_events=200000)

    benchmark.pedantic(run, setup=setup, rounds=7, warmup_rounds=1)

def test_bench_sdc1d_bindreplace_evolve(benchmark):
    """Benchmark evolve SDC1DBindReplace with N=8 bitcopy."""
    from rgrow.sdc import SDCStrand, SDCParams
    from rgrow.rgrow import SDC1DBindReplace
    from rgrow import State

    def make_bitcopy_params(N, input="0", conc=1e-7, cdl=10, sdl=20):
        strands = []

        match input:
            case "0":
                conc0 = 1e-7
            case "1":
                conc0 = 0
            case _:
                raise ValueError("Input must be '0' or '1'")

        strands.append(SDCStrand(conc0, "c0", "sc0", "c0*", "input0", color="blue"))

        for i in range(1,N):
            strands.append(SDCStrand(conc, "c0", f"sc{i}", "c0*", f"{i}_0", color="teal"))
            strands.append(SDCStrand(conc, "c1", f"sc{i}", "c1*", f"{i}_1", color="orange"))

        params = SDCParams(
            strands=strands,
            # glue_dg_s = (
            #     {"c0": mean_energies(cdl)} |
            #     {"c1": mean_energies(cdl)} |
            #     {f"sc{i}": mean_energies(sdl) for i in range(0,N) }
            # ),
            scaffold = [f"sc{i}*" for i in range(0,N)]
        )

        return params
    
    sys = SDC1DBindReplace(make_bitcopy_params(N=30, input="0"))

    def setup():
        state = State(
            (512, 30), kind="SquareCompact", tracking="None",
            n_tile_types=len(sys.tile_names),
        )
        sys.update_state(state)
        return (state,), {}
    
    def run(state):
        sys.evolve(state, for_events=100000)

    benchmark.pedantic(run, setup=setup, rounds=10, warmup_rounds=1)