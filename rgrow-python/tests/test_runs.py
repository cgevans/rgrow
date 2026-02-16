from rgrow import Tile, TileSet  # noqa: F841
import pytest


@pytest.mark.parametrize("canvas_type", ["square", "periodic"])
def test_full_simulation(canvas_type: str):
    # Make a simple sierpinski set:
    tiles = [  # noqa: F841
        Tile([0, "h", "v", 0]),
        Tile(["v", "e1", "v", 0]),
        Tile([0, "h", "e1", "h"]),
    ] + [
        Tile(
            ["e" + str(i1), "e" + str(i1 ^ i2), "e" + str(i1 ^ i2), "e" + str(i2)],
            name=f"t{i1}{i2}",
        )
        for i1 in [0, 1]
        for i2 in [0, 1]
    ]

    bonds = [("h", 2), ("v", 2)]

    options = {
        "seed": [(2, 2, 1)],
        "gse": 9.0,
        "gmc": 16.0,
        "canvas_type": canvas_type,
        "size": (64, 128),
    }

    ts = TileSet(tiles, bonds, **options)  # type: ignore

    sim, state = ts.create_system_and_state()

    sim.evolve(state, for_events=10000)

    assert state.ntiles > 60


@pytest.mark.parametrize("canvas_type", ["square", "periodic"])
def test_atam(canvas_type: str):
    # Make a simple sierpinski set:
    tiles = [
        Tile([0, "h", "v", 0]),
        Tile(["v", "e1", "v", 0]),
        Tile([0, "h", "e1", "h"]),
    ] + [
        Tile(
            ["e" + str(i1), "e" + str(i1 ^ i2), "e" + str(i1 ^ i2), "e" + str(i2)],
            name=f"t{i1}{i2}",
        )
        for i1 in [0, 1]
        for i2 in [0, 1]
    ]

    bonds = [("h", 2), ("v", 2)]

    options = {
        "seed": [(2, 2, 1)],
        "model": "atam",
        "canvas_type": canvas_type,
        "size": 100,
    }

    ts = TileSet(tiles, bonds, **options)  # type: ignore

    sim, state = ts.create_system_and_state()

    sim.evolve(state, for_events=3600)

    assert state.ntiles == 3601
