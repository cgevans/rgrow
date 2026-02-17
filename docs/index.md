# Rgrow

Rgrow is a fast, extensible simulator for Tile Assembly Models, primarily focused on kinetic models that require fast simulations of attachment and detachment events.  It is inspired by Xgrow, but seeks to be more extensible and easier and faster to use programmably.  It has Rust and Python interfaces, and a command line and UI interface similar to Xgrow.  It includes an implementation of forward flux sampling for nucleation rate calculations.

## Models

- [**kTAM** — Kinetic Tile Assembly Model](models/ktam.md)
- [**aTAM** — Abstract Tile Assembly Model](models/atam.md)
- [**kBlock** — kTAM with blockers](models/kblock.md)
- [**SDC** — 1D Scaffolded DNA Computers](models/sdc.md)

## Quick Links

- [Getting Started](getting-started.md) — Installation, basic usage
- [Examples](examples/index.md) — Jupyter notebook examples
- [API Reference](reference/index.md) — Python API documentation
- [Rust docs (docs.rs)](https://docs.rs/rgrow/) — Rust API documentation
- [GitHub](https://github.com/cgevans/rgrow) — Source code and issue tracker
