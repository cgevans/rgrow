"""Profiling driver for Python-side rgrow models.

Run under perf: perf record -F 997 --call-graph dwarf -o /tmp/p.perf -- \
    python rgrow-python/tests/_perf_drive.py [model] [n_events]
"""
import sys
import time
from rgrow import State, KTAM
from rgrow.kblock import KBlock, KBlockTile, KBlockParams
from rgrow.sdc import SDC, SDCParams, SDCStrand
from rgrow.rgrow import SDC1DBindReplace
import numpy as np

R_CONST = 1.98720425864083e-3


def run_kblock(n_events: int = 2_000_000):
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
    state = State((10, 256), kind="tube")
    state.canvas_view[::2, 5:50] = 1 << 4
    state.canvas_view[1::2, 5:50] = 2 << 4
    sys.update_all(state)
    t0 = time.perf_counter()
    sys.evolve(state, for_events=n_events)
    dt = time.perf_counter() - t0
    print(f"KBlock 10x256 tube (block-grow): {n_events} events in {dt:.2f}s = "
          f"{dt/n_events*1e9:.1f} ns/event, n_tiles={state.n_tiles}")


def _bitcopy_params(N, conc=1e-7, cdl=10, sdl=20):
    def mean_e(L=10):
        return (-1.405625 * (L - 1), -0.02201875 * (L - 1))
    strands = [SDCStrand(conc, "c0", "sc0", "c0*", "input0")]
    for i in range(1, N):
        strands.append(SDCStrand(conc, "c0", f"sc{i}", "c0*", f"{i}_0"))
        strands.append(SDCStrand(conc, "c1", f"sc{i}", "c1*", f"{i}_1"))
    glue = ({"c0": mean_e(cdl)} | {"c1": mean_e(cdl)} |
            {f"sc{i}": mean_e(sdl) for i in range(N)})
    return SDCParams(strands=strands, glue_dg_s=glue,
                     scaffold=[f"sc{i}*" for i in range(N)])


def run_sdc(n_events: int = 2_000_000):
    sys = SDC(_bitcopy_params(8))
    sys.temperature = 75.0
    state = State((1024, 8), kind="SquareCompact",
                  tracking="None", n_tile_types=len(sys.tile_names))
    sys.update_state(state)
    t0 = time.perf_counter()
    sys.evolve(state, for_events=n_events)
    dt = time.perf_counter() - t0
    print(f"SDC bitcopy N=8 1024 scaffolds: {n_events} events in {dt:.2f}s = "
          f"{dt/n_events*1e9:.1f} ns/event, n_tiles={state.n_tiles}")


def run_sdc_bindreplace(n_events: int = 2_000_000):
    sys = SDC1DBindReplace(_bitcopy_params(30))
    state = State((512, 30), kind="SquareCompact",
                  tracking="None", n_tile_types=len(sys.tile_names))
    sys.update_state(state)
    t0 = time.perf_counter()
    sys.evolve(state, for_events=n_events)
    dt = time.perf_counter() - t0
    print(f"SDC1DBindReplace N=30 512 scaffolds: {n_events} events in {dt:.2f}s "
          f"= {dt/n_events*1e9:.1f} ns/event, n_tiles={state.n_tiles}")


def run_ktam_ffs(target_size: int = 300, max_configs: int = 8000,
                  min_configs: int = 4000, canvas: int = 128):
    from rgrow.rgrow import TileSet
    from rgrow import FFSRunConfig
    ts = TileSet.from_file("examples/sierpinski.yaml")
    config = FFSRunConfig(
        target_size=target_size, max_configs=max_configs,
        min_configs=min_configs,
        canvas_size=(canvas, canvas), canvas_type="Periodic",
    )
    sys = KTAM.from_tileset(ts)
    t0 = time.perf_counter()
    res = sys.run_ffs(config)
    dt = time.perf_counter() - t0
    print(f"KTAM FFS sierpinski {canvas}x{canvas} target={target_size} "
          f"max_cfg={max_configs}: {dt:.2f}s, "
          f"nuc_rate={res.nucleation_rate:.3e}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    if target in ("kblock", "all"):
        run_kblock(n)
    if target in ("sdc", "all"):
        run_sdc(n)
    if target in ("sdcbr", "all"):
        run_sdc_bindreplace(n)
    if target in ("ffs", "all"):
        run_ktam_ffs()
