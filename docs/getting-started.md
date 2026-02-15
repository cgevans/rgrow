# Getting Started

## Installation

### From PyPI (recommended)

Releases are pushed, in both source and a variety of binary forms, to PyPI:

```bash
pip install rgrow
```

### From Git

To install directly from the repository:

```bash
pip install "git+https://github.com/cgevans/rgrow.git"
```

### Development Install

Check out the repository and build with [maturin](https://www.maturin.rs/):

```bash
git clone https://github.com/cgevans/rgrow.git
cd rgrow
maturin develop --release
```

For optimized builds targeting your CPU:

```bash
maturin develop --release -- -C target-cpu=native
```

## Quick Start

### Define a Tile Set

```python
import rgrow as rg

tileset = rg.TileSet(
    tiles=[
        rg.Tile(edges=["N", "E", "S", "W"], name="center", color="blue"),
    ],
    bonds=[("N", 1.0), ("E", 1.0), ("S", 1.0), ("W", 1.0)],
    gse=8.0,
    gmc=16.0,
    size=64,
    seed=(32, 32, "center"),
)
```

### Run a Simulation

```python
sim = rg.Simulation(tileset)
sim.add_state()
sim.evolve(for_events=10000)
```

### Visualize the Result

```python
import matplotlib.pyplot as plt

sys, state = tileset.create_system_and_state()
sys.evolve(state, for_events=10000)
sys.plot_canvas(sys, state)
plt.show()
```

### Forward Flux Sampling

rgrow includes an implementation of forward flux sampling (FFS) for computing nucleation rates:

```python
result = tileset.run_ffs(
    target_size=50,
    canvas_size=(64, 64),
)
print(f"Nucleation rate: {result.nucleation_rate}")
```

See the [FFS examples](examples/index.md#forward-flux-sampling) for detailed walkthroughs.

## Next Steps

- Browse the [Examples](examples/index.md) for interactive Jupyter notebooks
- Read the [API Reference](reference/index.md) for detailed documentation
