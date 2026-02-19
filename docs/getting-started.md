# Getting Started

## Forms and installation instructions

Rgrow is designed with several uses in mind:

- For most users, the Python library is the easiest way to use Rgrow.  It allows tile systems to be defined in Python code (or via tileset files), allows flexible interactions with simulations, interfaces with Polars and Numpy for data access and manipulation, and allows the GUI to be run.  It can conveniently be used through Jupyter notebooks as well.

- For exploring tile systems, the `rgrow` standalone executable operates like Xgrow (and can interpret many Xgrow tileset files), taking a declarative input file defining the tile system and displaying the simulation in a GUI.  The standalone executable is also included as a dependency of the Python package.

- For some users, interfacing directly with the core Rust library may make sense.

### Python package installation

The Python package has a number of pre-built wheels for Linux, MacOS, and Windows; in many cases, it should be easily installable via a command like `uv pip install rgrow` or `pip install rgrow`, depending on your preferred package manager.  The package includes both the Python library and the standalone executable.

When installing from source, you will need a Rust installation.  Installing the latest commit directly from the repository can be done with `uv pip install git+https://github.com/cgevans/rgrow`.  Note that this may take much longer than most python package installations, while the underlying Rust library is compiled.

### Standalone executable

The standalone `rgrow` executable can be installed in two ways:

- Installing the `rgrow` or `rgrow-cli` Python packages, which will install a pre-build wheel if possible.

- Installing `rgrow` with `cargo`, eg, `cargo install rgrow`.  This does not require Python at all, and will build the executable from source.

## Basic usage in Python

Rgrow is built around a series of *models*, within which *systems* can be defined, these in turn operate on *states*. 

- The *model* defines 
- The *system* 
- The *state* 

Rgrow has two ways of defining systems: the *tileset* interface, which was, and the *direct* interface.

For example, to use the tileset interface to create a basic XOR system in the aTAM, one might use

```python
from rgrow improt TileSet, Tile, Bond
tileset = TileSet(
    [
        Tile(name="S", edges=["null", "rb", "bb", "null"], color="purple", stoic=0),
        Tile(name="RB", edges=["null", "rb", "e1", "rb"], color="red"),
        Tile(name="BB", edges= ["bb", "e1", "bb", "null"], color="blue"),
        Tile(name="00", edges= ["e0", "e0", "e0", "e0"], color="teal"),
        Tile(name="10", edges= ["e1", "e1", "e1", "e0"], color="green"),
        Tile(name="01", edges= ["e0", "e1", "e1", "e1"], color="yellow"),
        Tile(name="11", edges= ["e1", "e0", "e0", "e1"], color="orange"),
    ],
    bonds=[Bond("rb", 2), Bond("bb", 2)],
    seed=[
        (0, 0, "S")
    ],
    threshold=2,
    model="aTAM",
    size=(32,32),
    canvas_type="SquareCompact",
)
```

Then, the tileset definition can be used to create a system and state:

```python
system, state = tileset.create_system_and_state()
```

And finally, the state can be evolved and plotted

```python
system.evolve(state, for_events=1023)
system.plot_canvas(state)
```
