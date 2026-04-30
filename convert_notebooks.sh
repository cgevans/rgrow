#!/usr/bin/env bash
# Convert Jupyter notebook symlinks in docs/examples/ to markdown for zensical.
# Run from the repository root.
set -euo pipefail

EXAMPLES_DIR="docs/examples"

for symlink in "$EXAMPLES_DIR"/*.ipynb; do
    [ -L "$symlink" ] || continue
    target=$(readlink -f "$symlink")
    basename=$(basename "$symlink" .ipynb)
    echo "Converting $target -> $EXAMPLES_DIR/$basename.md"
    jupyter nbconvert --to markdown --output-dir "$EXAMPLES_DIR" --output "$basename" "$target"
done
