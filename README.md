[![codecov](https://codecov.io/github/cgevans/rgrow/graph/badge.svg?token=GZLWKMQ2GZ)](https://codecov.io/github/cgevans/rgrow)
[![PyPI](https://img.shields.io/pypi/v/rgrow)](https://pypi.org/project/rgrow/)
[![Crates.io](https://img.shields.io/crates/v/rgrow)](https://crates.io/crates/rgrow)
[![docs.rs](https://docs.rs/rgrow/badge.svg)](https://docs.rs/rgrow)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://cgevans.github.io/rgrow/)

# Introduction

rgrow (which may change name in the future) is a fast, extensible simulator for Tile Assembly Models, primarily focused on kinetic models that require fast simulations of attachment and detachment events.  It is inspired by Xgrow, but seeks to be more extensible and easier and faster to use programmably.  It has Rust and Python interfaces, and a command line and UI interface similar to Xgrow.  It includes an implementation of forward flux sampling for nucleation rate calculations.

Python documentation is available at [https://cgevans.github.io/rgrow/](https://cgevans.github.io/rgrow/).

Rust documentation is available at [https://docs.rs/rgrow/](https://docs.rs/rgrow/).

For FFS examples, see the [examples/ffs](https://github.com/cgevans/rgrow/tree/main/examples/ffs) folder.

# Installation

## Python library

Releases are pushed, in both source and a variety of binary forms, to PyPI.  To install the latest release, use:

```bash
pip install rgrow[default]
```

To install directly from git using Pip:

```bash
pip install "git+https://github.com/cgevans/rgrow.git[default]"
```

or check out the repository, and use something like

```bash
maturin develop --release -- -C target-cpu=native
```

The UI used to have specific extra build instructions, but it should now just work.
