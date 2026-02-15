# rgrow

[![PyPI](https://img.shields.io/pypi/v/rgrow)](https://pypi.org/project/rgrow/)
[![Crates.io](https://img.shields.io/crates/v/rgrow)](https://crates.io/crates/rgrow)
[![docs.rs](https://docs.rs/rgrow/badge.svg)](https://docs.rs/rgrow)
[![codecov](https://codecov.io/github/cgevans/rgrow/graph/badge.svg?token=GZLWKMQ2GZ)](https://codecov.io/github/cgevans/rgrow)

rgrow is a fast, extensible simulator for Tile Assembly Models, primarily focused on kinetic models that require fast simulations of attachment and detachment events. It is inspired by [Xgrow](https://github.com/DNA-and-Natural-Algorithms-Group/xgrow), but seeks to be more extensible and easier and faster to use programmably.

It has **Rust** and **Python** interfaces, and a command line and UI interface similar to Xgrow. It includes an implementation of forward flux sampling for nucleation rate calculations.

## Supported Models

- [**kTAM** — Kinetic Tile Assembly Model](models/ktam.md)
- [**aTAM** — Abstract Tile Assembly Model](models/atam.md)
- [**kBlock** — kTAM with blockers](models/kblock.md)
- [**SDC** — 1D Strand Displacement Circuits](models/sdc.md)

## Quick Links

- [Getting Started](getting-started.md) — Installation and first simulation
- [Examples](examples/index.md) — Jupyter notebook walkthroughs
- [API Reference](reference/index.md) — Python API documentation
- [Rust docs (docs.rs)](https://docs.rs/rgrow/) — Rust API documentation
- [GitHub](https://github.com/cgevans/rgrow) — Source code and issue tracker
