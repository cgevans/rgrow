# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rgrow is a fast, extensible Tile Assembly Model simulator written in Rust with Python bindings (PyO3). It simulates kinetic attachment/detachment events, inspired by Xgrow. It includes forward flux sampling (FFS) for nucleation rate calculations, a GUI (iced), and a CLI.

## Build & Development Commands

Task automation uses [just](https://github.com/casey/just). Run `just` to list all recipes.

### Rust
```bash
cargo build -p rgrow           # Build (GUI enabled by default)
cargo test -p rgrow            # Run Rust tests
cargo clippy -p rgrow          # Lint
cargo bench --bench sierpinski  # Run a specific benchmark
```

### Python (requires venv at .venv)
```bash
maturin develop --uv                    # Build Python extension for dev
pytest rgrow-python/tests/ -v           # Run Python tests
pytest rgrow-python/tests/test_ktam.py  # Run a single test file
pytest -k "test_name" rgrow-python/tests/  # Run a specific test
```

### Combined
```bash
just test          # All tests (Rust + Python)
just lint          # clippy + ruff
just coverage      # Combined Rust + Python coverage (cargo-llvm-cov)
```

### Linting
- **Rust**: `cargo clippy` (denies all warnings in CI)
- **Python**: `ruff check rgrow-python/rgrow/` (100 char line length)
- Pre-commit hooks enforce both

## Workspace Structure

Cargo workspace with two members:
- **rgrow-rust/** — Core Rust library + CLI binary
- **rgrow-python/** — PyO3 bindings (`src/lib.rs`) and Python facade (`rgrow/`)

There is also `rgrow-cli/` (a thin CLI wrapper packaged separately for PyPI).

## Architecture

### Trait-Based Dispatch with `enum_dispatch`

The core pattern is zero-cost polymorphism via `enum_dispatch`. Traits define interfaces; enum variants hold concrete types; the `enum_dispatch` macro generates match-based dispatch.

**System layer** (`system/`):
- `System` trait — core simulation interface (evolve, rates, events)
- `SystemEnum` — dispatches to model implementations
- Split into focused modules: `core.rs` (traits), `dispatch.rs`, `gui.rs`, `analysis.rs`, `types.rs`

**Model implementations** (`models/`):
- `ktam.rs` — Kinetic TAM (primary model)
- `atam.rs` — Abstract TAM (deterministic, threshold-based)
- `sdc1d.rs` — Strand Displacement Cascade
- `sdc1d_bindreplace.rs` — SDC variant with bind/replace
- `kblock.rs` — Block assembly model with blockers
- `oldktam.rs` — Legacy KTAM for comparison
- `fission_base.rs` — Shared fission handling

**State layer** (`state.rs`):
- `State` trait + `StateEnum` — generic over canvas and tracker types
- `QuadTreeState<C: Canvas, T: StateTracker>` — concrete state with QuadTree rate store
- Macro `impl_clonable_state!` expands all canvas×tracker combinations

**Canvas types** (`canvas/`): `CanvasSquare`, `CanvasSquareCompact`, `CanvasPeriodic`, `CanvasTube`, `CanvasTubeDiagonals`

**Tracker types**: `NullStateTracker`, `OrderTracker`, `LastAttachTimeTracker`, `MovieTracker`, `PrintEventTracker`

### Data Flow
```
TileSet (parsed from file or built programmatically)
  → SystemEnum + StateEnum
    → Model-specific rate calculations
      → QuadTree RateStore (O(log n) event selection)
        → Canvas operations (tile placement/removal)
```

### Python Bindings
- `rgrow-rust/src/python.rs` — PyO3 module definition
- `rgrow-python/rgrow/__init__.py` — Python-facing API facade
- `rgrow-python/rgrow/rgrow.pyi` — type stubs
- Uses `numpy` for array interchange and `polars` for dataframe results

### GUI
- Built with `iced` (feature-gated behind `gui` feature, on by default)
- `gui/iced_gui.rs` — main GUI, `gui/shm_reader.rs` — shared memory IPC for subprocess mode

### Key Features (Cargo)
- `default = ["gui"]`
- `python` — enables PyO3/numpy/polars bindings
- `gui` — enables iced/tokio/image

## Conventions

- Unsafe canvas access uses typed wrappers (`PointSafe2`, `PointSafeHere`) with bounds checking at boundaries
- Error types use `thiserror` (`GrowError`, `RgrowError`, `ModelError`); converted to `pyo3::PyErr` for Python
- Serialization: `serde` with `serde-saphyr` (YAML), `serde_json`, `bincode`
- Parsing: `nom` for the Xgrow file format (`parser_xgrow.rs`)
- Parallelism: `rayon` for parallel iteration
- Python tests use `pytest` with `--benchmark-skip` by default; benchmarks via `pytest-benchmark`
