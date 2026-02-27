# rgrow justfile

# Default recipe
default:
    @just --list

# Run all tests (Rust + Python)
test: test-rust test-python

# Run Rust tests only
test-rust:
    cargo test -p rgrow

# Build Python extension (with venv)
build-python:
    source .venv/bin/activate && maturin develop --uv

# Run Python tests only (builds first)
test-python: build-python
    source .venv/bin/activate && pytest rgrow-python/tests/ -v

# Run Python benchmarks (builds first)
bench-python: build-python
    source .venv/bin/activate && pytest rgrow-python/tests/ --benchmark-only -v

# Run Python benchmarks and save JSON output
bench-python-json: build-python
    source .venv/bin/activate && pytest rgrow-python/tests/ --benchmark-only --benchmark-json benchmark-results.json -v

# Run Rust clippy
clippy:
    cargo clippy -p rgrow -- -D warnings

# Run Ruff on Python code
ruff:
    ruff check rgrow-python/rgrow/

# Run all lints
lint: clippy ruff

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
        -w /io/rgrow-cli \
        ghcr.io/pyo3/maturin:latest \
        build --release --out /io/dist \
        --interpreter /opt/python/cp313-cp313t/bin/python3 /opt/python/cp314-cp314t/bin/python3

# Build native macOS wheels for free-threaded Python (run on macOS)
build-freethreaded-macos:
    maturin build --release --out dist --interpreter python3.13t python3.14t
    cd rgrow-cli && maturin build --release --out ../dist --interpreter python3.13t python3.14t

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
    cd rgrow-cli && maturin build --release --out ../dist --interpreter python3.13t python3.14t

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
        -w /io/rgrow-cli \
        ghcr.io/pyo3/maturin:latest \
        build --release --out /io/dist \
        --interpreter /opt/python/cp313-cp313t/bin/python3 /opt/python/cp314-cp314t/bin/python3

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
    uv pip install ./rgrow-cli
    maturin develop --uv --profile dev
    pytest rgrow-python/tests -v
    cargo llvm-cov report --lcov --output-path coverage-python.lcov -p rgrow -p rgrow-python

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
    uv pip install ./rgrow-cli
    maturin develop --uv --profile dev
    pytest rgrow-python/tests -v
    # Generate combined coverage report
    cargo llvm-cov report --lcov --output-path coverage.lcov -p rgrow -p rgrow-python

# Generate HTML coverage report (run after coverage)
cov-html:
    #!/usr/bin/env bash
    set -euo pipefail
    source <(cargo llvm-cov show-env --export-prefix --no-cfg-coverage)
    export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
    cargo llvm-cov report --html -p rgrow -p rgrow-python

# Full coverage with HTML report
cov-full: coverage cov-html

# Serve docs locally with live reload
docs-serve:
    source .venv/bin/activate && mkdocs serve

# Build docs site
docs-build:
    source .venv/bin/activate && mkdocs build

# --- Profiling ---
# Prerequisites:
#   cargo install samply
#   cargo install flamegraph
#   sudo dnf install perf  (Fedora)

# Profile a Rust benchmark with samply (opens Firefox Profiler)
profile-bench-samply bench="sierpinski":
    #!/usr/bin/env bash
    set -euo pipefail
    cargo bench --bench {{bench}} --profile profiling --no-run
    bin=$(find target/profiling/deps -name '{{bench}}-*' -executable | head -1)
    echo "Profiling: $bin"
    samply record "$bin" --bench

# Profile a Rust benchmark with cargo-flamegraph (produces SVG)
profile-bench-flamegraph bench="sierpinski":
    cargo flamegraph --bench {{bench}} --profile profiling -o flamegraph-{{bench}}.svg -- --bench

# Profile a Python script with samply (builds with debug symbols first)
profile-python-samply script:
    #!/usr/bin/env bash
    set -euo pipefail
    source .venv/bin/activate
    maturin develop --profile profiling --uv
    samply record -- python {{script}}

# Profile a Python script with perf + flamegraph (builds with debug symbols first)
profile-python-flamegraph script:
    #!/usr/bin/env bash
    set -euo pipefail
    source .venv/bin/activate
    maturin develop --profile profiling --uv
    perf record -g --call-graph dwarf -- python {{script}}
    perf script | flamegraph > flamegraph-python.svg
