# rgrow justfile

# Default recipe
default:
    @just --list

# Run all tests (Rust + Python)
test: test-rust test-python

# Run Rust tests only
test-rust:
    cargo test -p rgrow

# Run Python tests (builds first)
test-python:
    source .venv/bin/activate && maturin develop --uv
    source .venv/bin/activate && pytest rgrow-python/tests/

# Run all linters
lint:
    cargo clippy -p rgrow
    ruff check rgrow-python/rgrow/

# Build manylinux wheels for free-threaded Python via Podman
build-freethreaded-linux:
    podman run --rm \
        -v "$(pwd)":/io \
        -w /io \
        ghcr.io/pyo3/maturin:latest \
        build --release --out dist \
        --interpreter /opt/python/cp313-cp313t/bin/python3 /opt/python/cp314-cp314t/bin/python3
    podman run --rm \
        -v "$(pwd)":/io \
        -w /io/rgrow-gui \
        ghcr.io/pyo3/maturin:latest \
        build --release --out /io/dist \
        --interpreter /opt/python/cp313-cp313t/bin/python3 /opt/python/cp314-cp314t/bin/python3

# Build native macOS wheels for free-threaded Python (run on macOS)
build-freethreaded-macos:
    maturin build --release --out dist --interpreter python3.13t python3.14t
    cd rgrow-gui && maturin build --release --out ../dist --interpreter python3.13t python3.14t

# Build free-threaded macOS wheels from a specific git ref using a worktree
build-freethreaded-macos-ref ref:
    #!/usr/bin/env bash
    set -euo pipefail
    worktree="/tmp/rgrow-build-macos-{{ref}}"
    echo "Creating worktree at $worktree for {{ref}}..."
    git worktree add "$worktree" "{{ref}}"
    trap 'echo "Copying wheels..."; cp "$worktree"/dist/*.whl dist/ 2>/dev/null || true; echo "Removing worktree..."; git worktree remove --force "$worktree"; echo "Done."' EXIT
    cd "$worktree"
    maturin build --release --out dist --interpreter python3.13t python3.14t
    cd rgrow-gui && maturin build --release --out ../dist --interpreter python3.13t python3.14t

# Build all free-threaded wheels (Linux via Podman + native macOS)
build-freethreaded: build-freethreaded-linux

# Build free-threaded Linux wheels from a specific git ref using a worktree
build-freethreaded-ref ref:
    #!/usr/bin/env bash
    set -euo pipefail
    worktree="/tmp/rgrow-build-{{ref}}"
    echo "Creating worktree at $worktree for {{ref}}..."
    git worktree add "$worktree" "{{ref}}"
    trap 'echo "Copying wheels..."; cp "$worktree"/dist/*.whl dist/ 2>/dev/null || true; echo "Removing worktree..."; git worktree remove --force "$worktree"; echo "Done."' EXIT
    podman run --rm \
        -v "$worktree":/io \
        -w /io \
        ghcr.io/pyo3/maturin:latest \
        build --release --out dist \
        --interpreter /opt/python/cp313-cp313t/bin/python3 /opt/python/cp314-cp314t/bin/python3
    podman run --rm \
        -v "$worktree":/io \
        -w /io/rgrow-gui \
        ghcr.io/pyo3/maturin:latest \
        build --release --out /io/dist \
        --interpreter /opt/python/cp313-cp313t/bin/python3 /opt/python/cp314-cp314t/bin/python3
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

# Serve docs locally with live reload
docs-serve:
    source .venv/bin/activate && mkdocs serve

# Build docs site
docs-build:
    source .venv/bin/activate && mkdocs build
