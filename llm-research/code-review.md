# rgrow Code Review

Focused review of simulation correctness, API usability, and performance.
Codebase snapshot: `cc835aa` (2026-03-23).

---

## 1. Simulation Correctness

### 1.1 CRITICAL: Bitwise OR causes unsafe access in dimer detachment rates

**Files:** `ktam.rs:1931`, `ktam.rs:1955`

The dimer detachment rate functions use bitwise OR (`|`) instead of short-circuit
logical OR (`||`) in a guard that precedes an unsafe canvas read:

```rust
// ktam.rs:1931 — dimer_s_detach_rate
if (!canvas.inbounds(p2)) | (unsafe { canvas.uv_p(p2) == 0 }) | self.is_seed(PointSafe2(p2))

// ktam.rs:1955 — dimer_e_detach_rate
if (!canvas.inbounds(p2)) | (unsafe { canvas.uv_p(p2) == 0 } | self.is_seed(PointSafe2(p2)))
```

With `|`, all three operands are evaluated regardless of earlier results. If
`canvas.inbounds(p2)` is false, the unsafe `canvas.uv_p(p2)` still executes on
an out-of-bounds point. This is **undefined behavior**.

**Fix:** Replace `|` with `||` so the unsafe read is short-circuited when the
point is out of bounds.

Note: the parenthesization also differs between the two lines (the second nests
`is_seed` inside the `uv_p` comparison's parens), which may indicate a second
subtle logic error.

---

### 1.2 ~~CRITICAL: Dimer attachment friend-set lookup uses wrong tile index~~ RESOLVED

Already fixed in current code — the friend-set lookups at `ktam.rs:2458–2498`
correctly use `tne`, `tee`, `tse` (east) and `tss`, `tse`, `tsw` (south).

---

### 1.3 ~~CRITICAL: Operator precedence bug in xgrow parser seed bounds check~~ FIXED

**File:** `parser_xgrow.rs:365–368`

`&&` bound tighter than `||`, so the canvas-type guard only applied to the
`y < 2` condition. Seeds at `x > size-3`, `y > size-3`, or `x < 2` were
silently clamped even on periodic canvases. Fixed by parenthesizing the full
bounds check before `&&`.

---

### 1.4 HIGH: Glue links parsed but silently discarded

**File:** `parser_xgrow.rs:329, 379`

Xgrow `g(i,j)=strength` arguments are parsed into a `gluelinks` vector:
```rust
XgrowArgs::GlueLink(g1, g2, v) => gluelinks.push((g1.into(), g2.into(), v)),
```

The `gluelinks` are returned from the parser but never applied to the bond
strength matrix anywhere in the code path. Users relying on glue links in their
xgrow input files will get silently incorrect simulations.

---

### 1.5 ~~HIGH: Debug `println!` in dimer attachment hot path~~ FIXED

Removed `println!` at `ktam.rs:2412` (hot path) and `ktam.rs:3343–3344` (test).
Lines 2467–2472 were already commented out.

---

### 1.6 HIGH: `perform_event` returns `f64::NAN` for energy change

**Files:** `ktam.rs:265` (via `system/core.rs`), `atam.rs:377`, `sdc1d.rs:1165`, `kblock.rs:965`

All model implementations return `f64::NAN` from `perform_event()` instead of
the actual energy change. This value propagates to `state.record_event()` and
into the `EnergyChangesTracker`. Any analysis that depends on per-event energy
changes (including the new `EnergyChangesTracker` histogram) receives garbage
data.

KTAM has `energy_change_for_event()` (lines 2232–2270) that computes the
correct value, but it's not wired into `perform_event`.

---

### 1.7 HIGH: `energy_contribution_from_point` panics on duple systems

**File:** `ktam.rs:2216`

```rust
if self.has_duples {
    todo!()
}
```

Any code path calling `state_energy()` on a KTAM system with duples will panic.

---

### 1.8 MODERATE: `calc_dimers` uses total concentrations, not free concentrations

**File:** `ktam.rs:918–948`

The FIXME comment (line 922) acknowledges this: `calc_dimers()` computes
formation rates using `tile_concs` (total) rather than `free_tile_concs`
(depleted). For FFS nucleation rate calculations, this can overestimate dimer
formation rates when depletion is significant.

---

### 1.9 MODERATE: `state_energy` ignores canvas boundaries

**File:** `ktam.rs:2274`

The function iterates over the full `nrows × ncols` range without respecting
the canvas's `inbounds()` region. On `CanvasSquare` (with 2-tile border), this
reads border tiles and computes spurious bond energies.

---

### 1.10 MODERATE: `FAKE_EVENT_RATE` leaks through for seeds and duple fakes

**File:** `ktam.rs:67, 1441–1453`

Seeds and fake duple parts return `1e-20` as their detachment rate instead of
exactly 0. This is described as an "ODD HACK" to "allow rate-based copying."
While the probability of selecting such an event is vanishingly small, it means:

- Seeds can theoretically be detached.
- The rate contributes (negligibly) to total rate, distorting time sampling.
- In very long simulations or extreme conditions, these events may fire.

---

### 1.11 MODERATE: Hardcoded thresholds in mismatch calculations

**Files:** `atam.rs:398`, `sdc1d.rs:1247`, `kblock.rs:1084`

Each model hardcodes a different mismatch threshold:
- ATAM: `self.threshold / 4.0` with `// FIXME: this is a hack`
- SDC: `-0.1` with `// Todo: fix this`
- KBlock: `KcalPerMol(-0.05)` with `// Todo: fix this`

These should be derived from model parameters or made configurable.

---

### 1.12 LOW: Non-deterministic `HashSet` iteration in SDC MFE

**File:** `sdc1d.rs:982`

The MFE (minimum free energy) algorithm iterates over a `HashSet`, whose order
is non-deterministic. In cases where multiple configurations have the same
energy, different runs may select different MFE structures, making results
irreproducible even with the same RNG seed.

---

## 2. API Usability

### 2.1 Python `.unwrap()` panics instead of raising exceptions

**File:** `python.rs:242, 249, 576, 1197, 1215`

Multiple places in the PyO3 bindings use `.unwrap()` on fallible operations,
which panics and crashes the Python process instead of raising a catchable
exception:

```rust
serde_json::to_writer(File::create(filename)?, &self.0).unwrap();  // 242
serde_json::from_reader(File::open(filename)?).unwrap();            // 249
o.map(|x| x.into_py_any(py).unwrap())                              // 576
```

These should use `.map_err(...)` to convert to `RgrowError`/`PyErr`.

---

### 2.2 Type stub missing `replace` parameter on `place_tile`

**File:** `rgrow.pyi` — all system class stubs

The Rust implementation of `py_place_tile` accepts a `replace: bool` parameter
(python.rs:1240), but the `.pyi` stubs only show:
```python
def place_tile(self, state: State, point: tuple[int, int], tile: int) -> float: ...
```

Users and IDEs won't discover the `replace` parameter.

---

### 2.3 `energy_histogram()` missing from type stubs

**File:** `rgrow.pyi`

The `State.energy_histogram()` method exists in Rust (python.rs:164–192) but
has no corresponding entry in the `.pyi` type stub file. Users relying on type
checkers won't see this method.

---

### 2.4 `canvas_view` safety warning incomplete

**File:** `__init__.py:658–666`

The docstring warns about memory safety but doesn't mention that **direct
modification of the returned array silently corrupts the internal rate cache**,
leading to incorrect simulation behavior with no error. This is the more likely
misuse scenario (vs. the GC lifetime issue that is mentioned).

---

### 2.5 `num_workers` parameter accepted but ignored

**File:** `rbffs.rs:73`

`RBFFSConfig.num_workers` is defined and accepted from Python, but never
actually used for controlling parallelism. Users who set it expect it to have
an effect.

---

### 2.6 Panics instead of errors for invalid inputs

Multiple places panic on bad input instead of returning `Result`/`PyErr`:

| Location | Trigger |
|----------|---------|
| `parser_xgrow.rs:357` | Size not Single variant |
| `tileset.rs:766` | SDC model in `create_system` |
| `rbffs.rs:960` | Unexpected `EvolveOutcome` |
| `rbffs.rs:1213` | Canvas size < 4 |
| `sdc1d.rs:556` | Missing scaffold friends |
| `ffs.rs:849` | Empty parent state list |

---

### 2.7 Docstring typo

**File:** `rgrow.pyi:769`

"Stop evelving each state when..." — should be "evolving."

---

## 3. Performance

### 3.1 HIGH: QuadTree `choose_point` panics on floating-point edge cases

**File:** `ratestore.rs:64–104`

The 4-way branching at each quadtree level uses sequential threshold
subtraction. If accumulated floating-point rounding error causes `threshold` to
remain slightly positive after all four children are exhausted, line 99 panics:

```rust
panic!("Failure in quadtree position finding: remaining threshold {threshold:?}, ...")
```

This is a hard crash with no recovery path. While rare in practice, it becomes
more likely with:
- Very large rate disparities (e.g., `1e-15` next to `1e5`)
- Many quadtree levels (large canvases)
- Very high event throughput (more chances to hit the edge case)

**Mitigation options:**
- Fall back to the last valid quadrant when the panic would trigger
- Use Kahan summation for the threshold subtraction chain
- Clamp `threshold` to `[0, total_rate]` before entering the loop

---

### 3.2 HIGH: `println!` in dimer hot path (repeated from §1.5)

Beyond correctness, the I/O serialization from `println!` in the dimer
attachment code dominates wall-clock time whenever dimers exist. Each `println!`
acquires a lock on stdout, flushes, and performs string formatting. In a tight
simulation loop, this can easily slow things down by 100–1000×.

---

### 3.3 MODERATE: O(all_dimers) search per point for dimer attachment

**File:** `ktam.rs:2452–2454, 2500`

The dimer attachment search iterates over **all** dimers (`ns_dimers`,
`we_dimers`) at every candidate point:

```rust
for &(t1, t2, dimer_conc) in &self.we_dimers { ... }
for &(t1, t2, dimer_conc) in &self.ns_dimers { ... }
```

Monomer attachment uses precomputed `friends_*` arrays to limit iteration to
only relevant tiles. The FIXME at line 2452 acknowledges this. For systems with
many tile types, this is O(n²) in the number of tiles per rate recalculation.

---

### 3.4 MODERATE: FFS is entirely sequential

**File:** `ffs.rs:686–770, 859–922`

`FFSRun::create()` generates all configurations at each surface level
sequentially. Each trial involves evolving one state, which is independent of
other trials. This is a natural target for rayon parallelism (as RBFFS already
does at `rbffs.rs:305`).

---

### 3.5 MODERATE: `astro_float::Consts` recreated on every call

**File:** `sdc1d.rs:941`

```rust
let mut cc = astro_float::Consts::new().expect("...");  // FIXME: don't keep making this
```

This high-precision constant set is allocated on every call to
`log_partition_function()`, which is called frequently during SDC rate
calculations.

---

### 3.6 LOW: Bitwise `&` used where `&&` would short-circuit

**Files:** `ktam.rs:1214, 1223, 1841`, `state.rs:573, 576`

Multiple places use `&` (bitwise AND) for boolean conditions instead of `&&`
(short-circuit AND). While correct for `bool` operands, this evaluates both
sides even when the first is false, adding unnecessary work in hot paths. The
compiler may optimize this, but it's not guaranteed.

---

### 3.7 LOW: QuadTree branch prediction

**File:** `ratestore.rs:74–100`

The 4-way if-chain in `choose_point` is data-dependent (the threshold value
determines which quadrant is selected). Over 7–10 levels, this produces ~10–30
hard-to-predict branches per event selection. A branchless approach (computing
the quadrant index arithmetically) could improve throughput on modern
out-of-order CPUs, though the improvement would likely be modest given that the
quadtree traversal is not the dominant cost.

---

### 3.8 LOW: `FAKE_EVENT_RATE` pollutes rate tree

**File:** `ktam.rs:67`

Every seed tile and fake duple part contributes `1e-20` to the total rate. In a
system with many seeds, this accumulates and distorts the exponential time
sampling. More importantly, each non-zero rate requires a quadtree update path,
adding O(log n) work that wouldn't exist if the rate were exactly 0.

---

## 4. Robustness / Technical Debt

### 4.1 Unsafe code audit summary

The `PointSafe2` / `PointSafeHere` newtype pattern is a good design for
encapsulating bounds-checked access. The main gaps:

| Location | Issue |
|----------|-------|
| `ktam.rs:1825` | `uget` on `duple_info` — no tile-index bounds proof |
| `ktam.rs:2139–2175` | `uget` on `energy_ns/we` — index validity assumed |
| `state.rs:797–824` | `copy_level_quad` — no bounds check on rate array indices |
| `canvas/tube_diagonals.rs:52–82` | Column shifts on wrap-around — needs proof |

### 4.2 FIXME/TODO density

Grepping for `FIXME`, `TODO`, `HACK`, and `todo!()` across the Rust source:

- **ktam.rs**: ~15 markers (concentrated in dimer code and energy calculations)
- **sdc1d.rs**: ~12 markers (rate correctness uncertainty, hardcoded values)
- **kblock.rs**: ~8 markers (blocker energy adjustment, fission)
- **ffs.rs / rbffs.rs**: ~5 markers (error handling)
- **state.rs**: ~3 markers (ratestore abstraction leaks)
- **parser_xgrow.rs**: 1 panic, 1 precedence bug

The highest-risk cluster is the dimer attachment/detachment code in ktam.rs,
which has active correctness bugs alongside multiple FIXME markers.

### 4.3 Test coverage gaps

The test suite has good coverage of basic KTAM and FFS workflows. Notable gaps:

- Dimer attachment correctness (the friend-set bug at §1.2 would be caught by a
  test that verifies dimer attachment rates match expected values)
- Extreme rate ratio scenarios for the quadtree
- Error paths in Python bindings (corrupt JSON, invalid coordinates)
- `CanvasTubeDiagonals` geometry correctness
- Energy tracking end-to-end (currently returns NaN everywhere)

---

## Priority Summary

| Priority | Issue | Section |
|----------|-------|---------|
| **P0 — Fix now** | Bitwise OR → UB in dimer detach | §1.1 |
| ~~P0~~ | ~~Friend-set index bug in dimer attach~~ (already fixed) | §1.2 |
| **P0 — Fix now** | Remove debug `println!` from dimer code | §1.5 / §3.2 |
| **P1 — Fix soon** | Parser operator precedence | §1.3 |
| **P1 — Fix soon** | Glue links silently discarded | §1.4 |
| **P1 — Fix soon** | Python `.unwrap()` → exceptions | §2.1 |
| **P1 — Fix soon** | QuadTree panic on FP edge case | §3.1 |
| **P2 — Plan** | Wire `energy_change_for_event` into `perform_event` | §1.6 |
| **P2 — Plan** | Precomputed dimer friend sets | §3.3 |
| **P2 — Plan** | Parallelize FFS | §3.4 |
| **P2 — Plan** | Type stub completeness | §2.2, §2.3 |
| **P3 — Backlog** | Duple `state_energy` / `energy_contribution` | §1.7 |
| **P3 — Backlog** | Mismatch threshold hardcoding | §1.11 |
| **P3 — Backlog** | `FAKE_EVENT_RATE` cleanup | §1.10 / §3.8 |
