from rgrow import State
from test_sdc import make_bitcopy, temperature_for_target_prob


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
    sys = create_sierpinski_tileset(pad_seed=True)

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
    sys = create_sierpinski_tileset(pad_seed=False)

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