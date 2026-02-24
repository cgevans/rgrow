# Project Analysis: rgrow

**rgrow** is a well-structured scientific computing project — a Tile Assembly Model simulator with Rust core, Python bindings, and GUI. Below are findings organized by severity.

---

## Critical / High Severity

### ~~1. CI Codecov filename mismatch (silent data loss)~~ FIXED

~~`.github/workflows/ci.yml` generates `coverage.lcov` (line 49) but the upload step references `coverage-rust.lcov` (line 52). Rust coverage is silently never uploaded to Codecov.~~

### ~~2. `place_tile` doesn't guard against overwriting existing tiles~~ FIXED

~~`rgrow-rust/src/models/ktam.rs:945` — `place_tile` will silently clobber an existing tile, corrupting tile counts, rates, and energy tracking. Currently only called during initialization, but the public API is unsafe.~~

Added `replace: bool` parameter. When `false`, returns `TilePlacementBlocked` error if the site is occupied. When `true`, properly removes existing tiles via `MonomerDetachment` events before placing. KTAM and ATAM overrides handle duple companion sites correctly. Python API defaults to `replace=True` for backward compatibility.

---

## Moderate Severity

### ~~3. Deprecated dependency: `serde_yaml`~~ FIXED

~~`serde_yaml 0.9.34` is archived/deprecated upstream. Should migrate to `serde_yml` or another YAML library.~~

Migrated to `serde-saphyr` (panic-free YAML parsing with good error reporting). Only deserialization was used (`from_str`, `from_reader`), so the migration was a drop-in replacement.

### 4. Dimer detachment energy calculation uncertainty

`rgrow-rust/src/models/ktam.rs:1750,1774` — The dimer detach rate formula has FIXMEs questioning whether bond energy double-counting/compensation is correct. Could produce wrong rates when `ChunkHandling::Detach` is enabled.

### 5. FFS size threshold hack with duples

`rgrow-rust/src/ffs.rs:1047` — Using `>=` instead of `==` for the first FFS surface crossing compensates for duple tile overshooting, but introduces a small systematic bias in nucleation rate estimates for duple systems.

### 6. OldKTAM duplication (~1,100 lines)

`rgrow-rust/src/models/oldktam.rs` duplicates significant logic from `ktam.rs` (energy matrices, friends structures, fission detection). It has `todo!()` stubs and is referenced in 11 files. Adds maintenance burden and bug-fix divergence risk.

### ~~7. Formatting drift~~ FIXED

~~No `rustfmt.toml` exists. `cargo fmt --check` found violations in ~10 files, with `sdc1d_bindreplace.rs` being the worst offender (indentation, trailing whitespace, long lines).~~

### 8. Clippy warnings (3 remaining)

~~17 in library code, 13 in benchmarks.~~ Auto-fixed 7 warnings. 3 remain:

- Private type `Glue` exposed through public fields (`sdc1d_bindreplace.rs:42-43`)
- `too_many_arguments` in `state.rs:230` and `python.rs:325` (8 args, limit 7)

---

## Low Severity

### ~~9. Unnecessary `unsafe impl Send/Sync` (3 locations)~~ FIXED

~~All three are redundant — the types already auto-derive `Send`/`Sync`:~~

- ~~`rgrow-rust/src/models/atam.rs:102-103`~~
- ~~`rgrow-rust/src/state.rs:676`~~
- ~~`rgrow-rust/benches/canvas_bounds.rs:14-15`~~

### ~~10. Dead code: `cffi.rs`~~ FIXED

~~`rgrow-rust/src/cffi.rs` (122 lines) is commented out in `lib.rs:41`. No active references.~~

### ~~11. 3 doc-build warnings~~ FIXED

~~Unescaped `<angle brackets>` in doc comments parsed as HTML:~~

- ~~`rgrow-rust/src/models/kblock.rs:361` — `Array2<Energy>`~~
- ~~`rgrow-rust/src/models/sdc1d.rs:1855` — `Vec<temperatures>`, `Vec<times>`~~

### 12. Duplicate dependency versions (transitive — cannot fix locally)

Both `rand` (0.8 + 0.9) and `thiserror` (1.x + 2.x) coexist in the dependency tree, increasing binary size. These are all from transitive dependencies: `rand 0.8` comes from `astro-float-num` and `zbus` (via iced); `thiserror 1.x` from `bpci`, `iced`, and transitive GUI deps. Cannot be resolved without upstream updates.

### 13. `state_energy` ignores canvas bounds

`rgrow-rust/src/models/ktam.rs:2085` — May produce wrong energy reports on toroidal/bounded canvases (diagnostic-only, doesn't affect simulation stepping).

### 14. ~60 FIXMEs and ~15 TODOs across codebase

Highest density in `ktam.rs` (18), `oldktam.rs` (11), and `sdc1d.rs` (8). Most are about incomplete edge-case handling rather than active bugs.

---

## Improvement Opportunities

### 15. Split `system.rs` (2,077 lines)

Contains at least 5 distinct concerns: event types, evolution bounds, the `System` trait, GUI/IPC integration (~490 lines), and statistical analysis (committor calculations, ~485 lines). Could be split into `system/core.rs`, `system/gui.rs`, `system/analysis.rs`, `system/types.rs`.

### 16. Missing `CLAUDE.md` / `CONTRIBUTING.md`

No developer onboarding documentation or coding conventions guide.

### 17. `old/` directory cleanup

`old/rgrow-gui/` and `old/rgrow-ipc/` are deprecation stubs not in the workspace. If already published to crates.io, the directory can be removed.

### 18. sdc1d.rs sign convention (likely false alarm)

The "Is there a minus missing here?" TODOs at lines 351 and 376 appear to be consistent with the internal sign convention (delta_G is negative for favorable bonds). A targeted unit test would confirm this.
