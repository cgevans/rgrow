# Rgrow

Rgrow is a fast, extensible simulator for Tile Assembly Models, primarily focused on kinetic models that require fast simulations of attachment and detachment events.  It is inspired by Xgrow, but seeks to be more extensible and easier and faster to use programmably.  It has Rust and Python interfaces, and a command line and UI interface similar to Xgrow.  It includes an implementation of forward flux sampling for nucleation rate calculations.

- [Getting Started](getting-started.md)
- [Examples](examples/index.md) 
- [API Reference](reference/index.md)
- [Rust docs (docs.rs)](https://docs.rs/rgrow/)
- [GitHub](https://github.com/cgevans/rgrow)

## Models

- [**kTAM** — Kinetic Tile Assembly Model](models/ktam.md)
- [**aTAM** — Abstract Tile Assembly Model](models/atam.md)
- [**kBlock** — kTAM with blockers](models/kblock.md)
- [**SDC** — 1D Scaffolded DNA Computers](models/sdc.md)
