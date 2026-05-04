> **This repository is a mirror.** Primary development happens at
> [codeberg.org/cge/rgrow](https://codeberg.org/cge/rgrow) —
> please file issues and pull requests there.

[![codecov](https://codecov.io/github/cgevans/rgrow/graph/badge.svg?token=GZLWKMQ2GZ)](https://codecov.io/github/cgevans/rgrow)
[![PyPI](https://img.shields.io/pypi/v/rgrow)](https://pypi.org/project/rgrow/)
[![Crates.io](https://img.shields.io/crates/v/rgrow)](https://crates.io/crates/rgrow)
[![docs.rs](https://docs.rs/rgrow/badge.svg)](https://docs.rs/rgrow)
[![Docs](https://img.shields.io/badge/docs-Codeberg%20Pages-blue)](https://cge.codeberg.page/rgrow/)

# Introduction

rgrow (which may change name in the future) is a fast, extensible simulator for Tile Assembly Models, primarily focused on kinetic models that require fast simulations of attachment and detachment events.  It is inspired by Xgrow, but seeks to be more extensible and easier and faster to use programmably.  It has Rust and Python interfaces, and a command line and UI interface similar to Xgrow.  It includes an implementation of forward flux sampling for nucleation rate calculations.

Python documentation is available at [https://cge.codeberg.page/rgrow/](https://cge.codeberg.page/rgrow/).

There is also an in-browser interface at [https://cge.codeberg.page/rgrow/app/](https://cge.codeberg.page/rgrow/app/), which runs rgrow directly in WebAssembly.

Rust documentation is available at [https://docs.rs/rgrow/](https://docs.rs/rgrow/).

For FFS examples, see the [examples/ffs](https://codeberg.org/cge/rgrow/src/branch/main/examples/ffs) folder.

# Installation

## Python library

Releases are pushed, in both source and a variety of binary forms, to PyPI.  To install the latest release, use:

```bash
pip install rgrow[default]
```

To install directly from git using Pip:

```bash
pip install "git+https://codeberg.org/cge/rgrow.git[default]"
```

or check out the repository, and use something like

```bash
maturin develop --release -- -C target-cpu=native
```

The UI used to have specific extra build instructions, but it should now just work.
