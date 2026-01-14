# Justfile for rgrow project

# Default recipe
default: test

# Run Rust tests only
test-rust:
    cargo test -p rgrow

# Build Python extension (with venv)
build-python:
    source .venv/bin/activate && maturin develop --uv

# Run Python tests only (assumes extension is built)
test-python: build-python
    source .venv/bin/activate && pytest py-rgrow/tests -v

# Run all tests (Rust + Python)
test: test-rust test-python

# Run Rust clippy
clippy:
    cargo clippy -p rgrow

# Run Ruff on Python code
ruff:
    ruff check py-rgrow/rgrow/

# Run all lints
lint: clippy ruff

# Clean coverage data
cov-clean:
    cargo llvm-cov clean --workspace

# Run tests with coverage (Rust tests only)
cov-rust:
    #!/usr/bin/env bash
    set -euo pipefail
    source <(cargo llvm-cov show-env --export-prefix --no-cfg-coverage)
    export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
    export CARGO_INCREMENTAL=1
    cargo llvm-cov clean --workspace
    cargo test -p rgrow
    cargo llvm-cov report --lcov --output-path coverage-rust.lcov -p rgrow

# Run tests with coverage (Python tests, collecting Rust coverage)
cov-python:
    #!/usr/bin/env bash
    set -euo pipefail
    source <(cargo llvm-cov show-env --export-prefix --no-cfg-coverage)
    export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
    export CARGO_INCREMENTAL=1
    cargo llvm-cov clean --workspace
    source .venv/bin/activate
    maturin develop --uv --profile dev
    pytest py-rgrow/tests -v
    cargo llvm-cov report --lcov --output-path coverage-python.lcov -p rgrow -p py-rgrow

# Run all tests with combined Rust coverage (both Rust and Python tests)
coverage:
    #!/usr/bin/env bash
    set -euo pipefail
    source <(cargo llvm-cov show-env --export-prefix --no-cfg-coverage)
    export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
    export CARGO_INCREMENTAL=1
    cargo llvm-cov clean --workspace
    # Run Rust tests
    cargo test -p rgrow
    # Build and run Python tests
    source .venv/bin/activate
    maturin develop --uv --profile dev
    pytest py-rgrow/tests -v
    # Generate combined coverage report
    cargo llvm-cov report --lcov --output-path coverage.lcov -p rgrow -p py-rgrow

# Generate HTML coverage report (run after coverage)
cov-html:
    #!/usr/bin/env bash
    set -euo pipefail
    source <(cargo llvm-cov show-env --export-prefix --no-cfg-coverage)
    export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
    cargo llvm-cov report --html -p rgrow -p py-rgrow
    echo "Coverage report generated in target/llvm-cov/html/"

# Full coverage with HTML report
cov-full: coverage cov-html
