from rgrow import State
from test_sdc import make_bitcopy, temperature_for_target_prob
from test_sierpinski import create_sierpinski_tileset


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
    """Benchmark evolve SDC simple sierpinski with Square canvas."""
    sys = create_sierpinski_tileset(pad_seed=True).create_system()

    def setup():
        state = State(
            (1020, 1020), kind="Square", tracking="None",
            n_tile_types=len(sys.tile_names),
        )
        sys.update_state(state)
        return (state,), {}

    def run(state):
        sys.evolve(state, for_events=10000)

    benchmark.pedantic(run, setup=setup, rounds=5, warmup_rounds=1)

def test_bench_evolve_sierpinski_squarecompact(benchmark):
    """Benchmark evolve SDC simple sierpinski with SquareCompact canvas."""
    sys = create_sierpinski_tileset(pad_seed=False).create_system()

    def setup():
        state = State(
            (1020, 1020), kind="SquareCompact", tracking="None",
            n_tile_types=len(sys.tile_names),
        )
        sys.update_state(state)
        return (state,), {}

    def run(state):
        sys.evolve(state, for_events=10000)

    benchmark.pedantic(run, setup=setup, rounds=5, warmup_rounds=1)

def test_bench_sdc1d_bindreplace_evolve(benchmark):
    """Benchmark evolve SDC1DBindReplace with N=8 bitcopy."""
    from rgrow.sdc import SDC, SDCStrand, SDCParams
    from rgrow.rgrow import SDC1DBindReplace
    from rgrow import State

    def make_bitcopy_params(N, input="0", conc=1e-7, cdl=10, sdl=20):
        strands = []

        match input:
            case "0":
                conc0 = 1e-7
                conc1 = 0
            case "1":
                conc0 = 0
                conc1 = 1e-7
            case _:
                raise ValueError("Input must be '0' or '1'")

        strands.append(SDCStrand(conc0, "c0", "sc0", "c0*", "input0", color="blue"))
        # strands.append(SDCStrand(conc1, "c1", "sc0", "c1*", "input1", color="red"))

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