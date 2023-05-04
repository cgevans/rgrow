# Introduction

rgrow (which may change name in the future) is a fast, extensible simulator for Tile Assembly Models, primarily focused on kinetic models that require fast simulations of attachment and detachment events.  It is inspired by Xgrow, but seeks to be more extensible and easier and faster to use programmably.  It has Rust and Python interfaces, and a command line and UI interface similar to Xgrow.

# Installation

## Python library

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

```bash
cargo install rgrow
```