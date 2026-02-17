import rgrow as rg

def create_sierpinski_tileset(pad_seed=False):
    tileset = rg.TileSet(
        [
            rg.Tile(name="S", edges=["null", "rb", "bb", "null"], color="purple", stoic=0),
            rg.Tile(name="RB", edges=["null", "rb", "e1", "rb"], color="red"),
            rg.Tile(name="BB", edges= ["bb", "e1", "bb", "null"], color="blue"),
            rg.Tile(name="00", edges= ["e0", "e0", "e0", "e0"], color="teal"),
            rg.Tile(name="10", edges= ["e1", "e1", "e1", "e0"], color="green"),
            rg.Tile(name="01", edges= ["e0", "e1", "e1", "e1"], color="yellow"),
            rg.Tile(name="11", edges= ["e1", "e0", "e0", "e1"], color="orange"),
        ],
        bonds=[rg.Bond("rb", 2), rg.Bond("bb", 2)],
        seed=[
            (2 if pad_seed else 0, 2 if pad_seed else 0, "S")
        ],
        gmc=16,
        gse=8.6,
        size=(32,32),
        canvas_type="SquareCompact",
    )
    return tileset

def test_consistent_atam():
    ts = create_sierpinski_tileset()
    ts.model = "aTAM"
    ts.threshold = 2
    sys = ts.create_system()
    state = rg.State((32,32), kind="SquareCompact", tracking="None", n_tile_types=len(ts.tiles))
    sys.update_state(state)
    sys.evolve(state, for_events=1023)

    state2 = rg.State((32,32), kind="SquareCompact", tracking="None", n_tile_types=len(ts.tiles))
    sys.update_state(state2)
    sys.evolve(state2, for_events=1023)

    assert (state.canvas_view == state2.canvas_view).all()

    assert state2.total_rate == 0
    