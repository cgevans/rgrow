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
    source .venv/bin/activate && pytest py-rgrow/tests/

# Run all linters
lint:
    cargo clippy -p rgrow
    ruff check py-rgrow/rgrow/

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
