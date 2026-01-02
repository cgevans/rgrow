# rgrow justfile

# Default recipe
default:
    @just --list

# Build Python documentation
docs:
    uv sync --group docs
    uv run sphinx-apidoc -f -M -o py-rgrow/docs/source/api py-rgrow/rgrow
    cd py-rgrow/docs && uv run sphinx-build -M html source build

# Build wheels for the current platform
wheels:
    maturin build --release --out dist
    cd rgrow-gui && maturin build --release --out ../dist

# Build sdist
sdist:
    maturin sdist --out dist
    cd rgrow-gui && maturin sdist --out ../dist

# Run Rust tests with coverage
test-rust-cov:
    #!/usr/bin/env bash
    set -euo pipefail
    source <(cargo llvm-cov show-env --export-prefix --no-cfg-coverage)
    export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
    export CARGO_INCREMENTAL=1
    cargo llvm-cov clean --workspace
    cargo test -p rgrow
    cargo llvm-cov --no-run --lcov --output-path coverage.lcov -p rgrow -p py-rgrow

# Run Python tests with coverage
test-python-cov:
    uv run maturin develop --uv
    uv run pytest -vvv --cov=py-rgrow/rgrow --cov-report xml py-rgrow/tests

# Run all tests with coverage
test-cov: test-rust-cov test-python-cov

# Run Rust tests (no coverage)
test-rust:
    cargo test -p rgrow

# Run Python tests (no coverage)
test-python:
    uv run maturin develop --uv
    uv run pytest py-rgrow/tests

# Run all tests
test: test-rust test-python

# Run clippy
clippy:
    cargo clippy -p rgrow

# Run ruff
ruff:
    uvx ruff check py-rgrow/rgrow

# Run ty type checker
ty:
    uvx ty check py-rgrow/rgrow

# Run all lints
lint: clippy ruff

# Development build
dev:
    uv run maturin develop --uv

# Clean build artifacts
clean:
    rm -rf dist target/wheels
    cd py-rgrow/docs && rm -rf build

