# Introduction

rgrow (which may change name in the future) is a fast, extensible simulator for Tile Assembly Models, primarily focused on kinetic models that require fast simulations of attachment and detachment events.  It is inspired by Xgrow, but seeks to be more extensible and easier and faster to use programmably.  It has Rust and Python interfaces, and a command line and UI interface similar to Xgrow.  It includes an implementation of forward flux sampling for nucleation rate calculations.

Python documentation is available at [https://rgrow.readthedocs.io/en/latest/](https://rgrow.readthedocs.io/en/latest/).

Rust documentation is available at [https://docs.rs/rgrow/](https://docs.rs/rgrow/).

For FFS examples, see the [examples/ffs](https://github.com/cgevans/rgrow/tree/main/examples/ffs) folder.

# Installation

## Python library

Releases are pushed, in both source and a variety of binary forms, to PyPI.  To install the latest release, use:

```bash
pip install rgrow
```

To install directly from git using Pip:

```bash
pip install "git+https://github.com/cgevans/rgrow.git#subdirectory=py-rgrow"
```

or check out the repository, and use

```bash
cd py-rgrow
maturin develop --release -- -C target-cpu=native
```

The FLTK-based UI is usable within Python if you have the correct dependencies, and activate the "ui" feature:

```bash
cd py-rgrow
maturin develop --release -F ui -- -C target-cpu=native
```

## Rust / CLI

Note that the rust version currently requires Rust nightly.  This should change soon.

```bash
cargo install rgrow
```