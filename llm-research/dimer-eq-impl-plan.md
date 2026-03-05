# Implementation Plan: Dimer Equilibrium via equiconc

Based on the research in [dimer-eq.md](dimer-eq.md).

## Overview

Three phases, each independently useful and testable:

1. **Accurate dimer reporting** — equiconc replaces the naive formula in `calc_dimers()`, with depletion warnings
2. **Dimer attachment + depletion** — `ChunkHandling::Equilibrium` enables dimer attachment events with depleted monomer concentrations
3. **KBlock integration** — same treatment for the KBlock model (direct, since it already uses physical units)

---

## Phase 1: Accurate dimer reporting

### Step 1.1: Add equiconc dependency

**File**: `rgrow-rust/Cargo.toml`

Add under `[dependencies]`:
```toml
equiconc = { path = "../equiconc" }
```

Once equiconc is published to crates.io, switch to a version specifier.

### Step 1.2: Add cached equilibrium state to KTAM

**File**: `rgrow-rust/src/models/ktam.rs`

Add fields to `KTAM` struct (after existing fields, before the "calculated stuff" section around line 120):

```rust
/// Cached equilibrium dimer concentrations from equiconc, keyed by dimer name.
/// Recomputed in update_system(). Present whenever there are any dimers.
#[serde(skip)]
dimer_eq_concs: Vec<(Tile, Tile, Orientation, f64)>,  // (t1, t2, orient, eq_conc)

/// Free monomer concentrations after dimer depletion.
/// Only used when depletion is active (ChunkHandling::Equilibrium).
/// Always computed (for warnings), but only applied to rates when depletion is on.
#[serde(skip)]
free_tile_concs: Array1<Conc>,
```

Initialize both in `new_sized()` (with `free_tile_concs` cloned from `tile_concs`) and in deserialization fixup.

### Step 1.3: Implement `compute_dimer_equilibrium()`

**File**: `rgrow-rust/src/models/ktam.rs`

New private method on `KTAM`:

```rust
fn compute_dimer_equilibrium(&self) -> Result<equiconc::Equilibrium, GrowError> {
    // T = 1/R makes ΔG° = -energy_dimensionless, producing K = exp(energy).
    let t_ref = 1.0 / equiconc::R;
    let mut sys = equiconc::System::new().temperature(t_ref);

    // Add each real tile as a monomer (skip tile 0 = empty, skip fake duple parts)
    for t in 1..self.tile_concs.len() {
        if self.tile_concs[t] > 0.0 && self.should_be_counted[t] {
            sys = sys.monomer(&self.tile_names[t], self.tile_concs[t]);
        }
    }

    // Add NS dimers
    for ((t1, t2), e) in self.energy_ns.indexed_iter() {
        if *e > 0.0 && t1 > 0 && t2 > 0 {
            let conc1 = self.get_effective_concentration(t1 as Tile);
            let conc2 = self.get_effective_concentration(t2 as Tile);
            if conc1 <= 0.0 || conc2 <= 0.0 { continue; }

            let name1 = /* effective tile name for t1 (resolve fake duples) */;
            let name2 = /* effective tile name for t2 */;

            let delta_g = -*e;
            // No homodimer symmetry correction needed: kTAM energies are for
            // distinguishable lattice positions (north/south), and equiconc
            // does not apply symmetry factors either. Both give K = exp(E).

            let dimer_name = format!("{}_{}_NS", name1, name2);
            sys = sys.complex(&dimer_name, &[(name1, 1), (name2, 1)], delta_g);
        }
    }

    // Same for WE dimers (with "_WE" suffix to avoid name collisions)

    sys.equilibrium().map_err(|e| /* convert to GrowError */)
}
```

**Key details**:
- Tile names must be unique across the monomer list. They already are (enforced by TileSet parsing).
- Fake duple parts (where `!should_be_counted[t]`) are skipped as monomers. Their effective concentration maps to the real tile via `get_effective_concentration()`.
- A tile pair can form both NS and WE dimers — these are distinct complexes with separate names.
- The homodimer symmetry correction is needed because equiconc treats composition `[("A", 2)]` as an ordered complex. Verify this is correct against the existing kTAM convention by checking that the naive formula `[A]^2 * exp(E)` matches equiconc's `[A_A]` for the undepleted case (add a test for this).

### Step 1.4: Call from `update_system()` and cache results

**File**: `rgrow-rust/src/models/ktam.rs`, in `update_system()` (after energy matrices and friends lists are built, ~line 1300)

```rust
// Compute dimer equilibrium and cache results
self.dimer_eq_concs.clear();
self.free_tile_concs = self.tile_concs.clone();

// Only run if there are any non-zero energies (skip trivial case)
let has_dimers = self.energy_ns.iter().any(|e| *e > 0.0)
    || self.energy_we.iter().any(|e| *e > 0.0);

if has_dimers {
    match self.compute_dimer_equilibrium() {
        Ok(eq) => {
            // Cache free monomer concentrations
            for t in 1..self.tile_concs.len() {
                if self.tile_concs[t] > 0.0 && self.should_be_counted[t] {
                    self.free_tile_concs[t] = eq.concentration(&self.tile_names[t])
                        .unwrap_or(self.tile_concs[t]);
                }
            }

            // Cache dimer concentrations (for calc_dimers and dimer attachment)
            // Iterate energy matrices, look up each dimer by name
            for ((t1, t2), e) in self.energy_ns.indexed_iter() {
                if *e > 0.0 && t1 > 0 && t2 > 0 {
                    let name = format!(/* same as in compute_dimer_equilibrium */);
                    let conc = eq.concentration(&name).unwrap_or(0.0);
                    self.dimer_eq_concs.push((t1 as Tile, t2 as Tile, Orientation::NS, conc));
                }
            }
            // Same for WE

            // Depletion warning (when not using Equilibrium chunk handling)
            if self.chunk_handling != ChunkHandling::Equilibrium {
                for t in 1..self.tile_concs.len() {
                    if self.tile_concs[t] > 0.0 && self.should_be_counted[t] {
                        let frac = 1.0 - self.free_tile_concs[t] / self.tile_concs[t];
                        if frac > 0.1 {
                            log::warn!(
                                "Tile '{}' has {:.0}% depletion from dimerization. \
                                 Consider chunk_handling: equilibrium.",
                                self.tile_names[t], frac * 100.0
                            );
                        }
                    }
                }
            }
        }
        Err(e) => {
            log::warn!("Failed to compute dimer equilibrium: {e}. Using naive concentrations.");
            // dimer_eq_concs stays empty; free_tile_concs stays = tile_concs
        }
    }
}
```

### Step 1.5: Rewrite `calc_dimers()` to use cached results

**File**: `rgrow-rust/src/models/ktam.rs`, replace `calc_dimers()` body (~line 898)

```rust
fn calc_dimers(&self) -> Result<Vec<DimerInfo>, GrowError> {
    let mut dvec = Vec::new();

    for &(t1, t2, ref orientation, eq_conc) in &self.dimer_eq_concs {
        let conc1 = self.get_effective_concentration(t1);
        let conc2 = self.get_effective_concentration(t2);
        let biconc: MolarSq = (conc1 * conc2).into();

        dvec.push(DimerInfo {
            t1,
            t2,
            orientation: orientation.clone(),
            formation_rate: Into::<PerMolarSecond>::into(self.kf) * biconc,
            equilibrium_conc: Molar::new(eq_conc),
        });
    }

    // Fallback: if equiconc wasn't run (no dimers cached), use old formula
    if dvec.is_empty() && self.dimer_eq_concs.is_empty() {
        // ... keep old code path as fallback for edge cases
    }

    Ok(dvec)
}
```

Note: `formation_rate` remains `kf * [t1] * [t2]` (the second-order rate of encounters). This is the rate at which the two monomers would collide, independent of equilibrium. Only `equilibrium_conc` changes to use equiconc results.

### Step 1.6: Add `log` dependency if not present

Check if `log` crate is already in dependencies. If not, add it. (The `warn!` macro needs it.)

### Step 1.7: Tests

**File**: New test cases in `rgrow-rust/src/models/ktam.rs` (in the `#[cfg(test)]` block at the bottom)

1. **Consistency test**: For weak binding (low `g_se`), verify equiconc dimer concentrations match the naive formula within a tolerance. This confirms the energy mapping is correct.

2. **Homodimer test**: Two identical tiles forming a homodimer. Verify equiconc result matches the analytical formula `[AA] = [A_free]^2 * exp(E)` (no symmetry factor — kTAM positions are distinguishable, and equiconc doesn't add one either).

3. **Depletion test**: Strong binding (`g_se` = 20), verify `free_tile_concs[t] < tile_concs[t]` significantly.

4. **Mass conservation test**: Verify `free_tile_concs[t] + sum_of_dimers_containing_t == tile_concs[t]` for each tile.

5. **Parameter change test**: Modify `g_se` via `set_param()`, verify `dimer_eq_concs` and `free_tile_concs` are recomputed (not stale).

6. **No-dimer test**: System with no matching glues. Verify `free_tile_concs == tile_concs` and `dimer_eq_concs` is empty.

**File**: `rgrow-python/tests/test_ktam.py`

7. Python-side test: Create a system, check `calc_dimers()` returns sensible equilibrium concentrations, check they change when `g_se` is modified.

---

## Phase 2: Dimer attachment under ChunkHandling::Equilibrium

### Step 2.1: Add `effective_monomer_conc()` helper

**File**: `rgrow-rust/src/models/ktam.rs`

```rust
/// Returns the monomer concentration to use for attachment rates.
/// When ChunkHandling::Equilibrium is active, returns the depletion-adjusted
/// free concentration; otherwise returns the total concentration.
#[inline]
fn effective_monomer_conc(&self, t: usize) -> Conc {
    match self.chunk_handling {
        ChunkHandling::Equilibrium => self.free_tile_concs[t],
        _ => self.tile_concs[t],
    }
}
```

### Step 2.2: Use depleted concentrations in monomer attachment

**File**: `rgrow-rust/src/models/ktam.rs`

In `_find_monomer_attachment_possibilities_at_point()` (~line 1635):

```rust
// Change:
let rate = self.kf * self.tile_concs[t as usize];
// To:
let rate = self.kf * self.effective_monomer_conc(t as usize);
```

Also in `total_monomer_attachment_rate_at_point()` if it exists separately (check — it may just call the same function with `just_calc = true`).

### Step 2.3: Build dimer attachment lookup table

**File**: `rgrow-rust/src/models/ktam.rs`

Add a field to `KTAM`:
```rust
/// Pre-computed dimer attachment info: (t1, t2, dimer_eq_conc).
/// Grouped by the "friends" pattern: for a given neighbor configuration,
/// which dimers could attach?
#[serde(skip)]
ns_dimer_attachments: Vec<(Tile, Tile, Conc)>,
#[serde(skip)]
we_dimer_attachments: Vec<(Tile, Tile, Conc)>,
```

Populate in `update_system()` from `dimer_eq_concs` (filter to only dimers with `eq_conc > 0`).

### Step 2.4: Implement `total_dimer_attachment_rate_at_point()`

**File**: `rgrow-rust/src/models/ktam.rs`

New method. For an empty point, iterate over possible dimer attachments where one tile of the dimer goes at this point and the other goes at an adjacent empty point. The rate for each is `kf * dimer_eq_conc`.

```rust
fn total_dimer_attachment_rate_at_point<S: State>(&self, state: &S, p: PointSafe2) -> Rate64 {
    let mut total = 0.0;

    // This point is the "north" tile of a NS dimer (partner goes south)
    let ps = state.move_sa_s(p);
    if state.inbounds(ps.0) && state.tile_at_point(PointSafe2(ps.0)) == 0 {
        let tn = state.tile_to_n(p);
        let tw = state.tile_to_w(p);
        let te = state.tile_to_e(p);
        // neighbors of south position
        let ps2 = PointSafe2(ps.0);
        let ts_s = state.tile_to_s(ps2);
        let tw_s = state.tile_to_w(ps2);
        let te_s = state.tile_to_e(ps2);

        for &(t1, t2, dimer_conc) in &self.ns_dimer_attachments {
            // t1 goes at p (north), t2 goes at ps (south)
            // Check t1 is compatible with p's neighbors (excluding south, which is empty/t2)
            // Check t2 is compatible with ps's neighbors (excluding north, which is t1)
            if self.tile_fits_neighbors_for_dimer(t1, tn, tw, te)
                && self.tile_fits_neighbors_for_dimer_s(t2, ts_s, tw_s, te_s)
            {
                total += self.kf * dimer_conc;
            }
        }
    }

    // This point is the "west" tile of a WE dimer (partner goes east)
    // ... analogous ...

    // This point is the "south" tile of a NS dimer (partner goes north)
    // ... analogous (iterate ns_dimer_attachments with reversed roles) ...

    // This point is the "east" tile of a WE dimer (partner goes west)
    // ... analogous ...

    total
}
```

**Important**: A dimer can attach with either tile at the current point. For a NS dimer (t1, t2), point p could be where t1 goes (with t2 south) or where t2 goes (with t1 north). Both must be considered. However, be careful not to double-count: when the rate store processes the *neighbor* point, it will find the same dimer from the other side. Since `event_rate_at_point` is called per-point and rates are stored per-point, each point independently contributes its share. The dimer attachment event is triggered from whichever point is selected, so the total rate across both points is `2 * kf * [dimer]` for the pair, which is correct (either end can be the "landing" point).

**Simplification**: Rather than checking full neighbor compatibility (which is what `_find_monomer_attachment_possibilities_at_point` does with the friends lists), the dimer attachment check could use the same friends-list infrastructure. A dimer (t1, t2) NS can attach at point p (as t1) if t1 is in the friends lists for p's neighbors AND point south is empty AND t2 is in the friends lists for the south point's neighbors.

### Step 2.5: Implement `choose_dimer_attachment_at_point()`

**File**: `rgrow-rust/src/models/ktam.rs`

Similar structure to `_find_monomer_attachment_possibilities_at_point()` but iterates dimer possibilities:

```rust
fn choose_dimer_attachment_at_point<S: State>(
    &self,
    state: &S,
    p: PointSafe2,
    mut acc: Rate64,
) -> (bool, Rate64, Event, Rate64) {
    // Try NS dimers with t1 at p, t2 at south
    // ... check empty, check compatibility, subtract rate from acc ...
    // If selected, return Event::PolymerAttachment(vec![(p, t1), (ps, t2)])

    // Try WE dimers with t1 at p, t2 at east
    // ...

    // Try NS dimers with t2 at p, t1 at north
    // ...

    // Try WE dimers with t2 at p, t1 at west
    // ...

    (false, acc, Event::None, f64::NAN)
}
```

Use `Event::PolymerAttachment(Vec<(PointSafe2, Tile)>)` — this variant already exists and `perform_event` already handles it, including duple expansion. No new Event variant needed.

### Step 2.6: Wire into `event_rate_at_point()` and `choose_event_at_point()`

**File**: `rgrow-rust/src/models/ktam.rs`

In `event_rate_at_point()` (~line 310), replace the `ChunkHandling::Equilibrium` branch:

```rust
ChunkHandling::Equilibrium => {
    if t.nonzero() {
        self.monomer_detachment_rate_at_point(state, p).to_per_second()
            + self.chunk_detach_rate(state, p, t).to_per_second()
    } else {
        self.total_monomer_attachment_rate_at_point(state, p).to_per_second()
            + self.total_dimer_attachment_rate_at_point(state, p).to_per_second()
    }
}
```

In `choose_event_at_point()` (~line 324), add dimer attachment as a fallback after monomer attachment:

```rust
fn choose_event_at_point<S: State>(&self, state: &S, p: PointSafe2, acc: PerSecond) -> (Event, f64) {
    match self.choose_detachment_at_point(state, p, Rate64::from_per_second(acc)) {
        (true, _, event, rate) => (event, rate),
        (false, acc, _, _) => match self.choose_attachment_at_point(state, p, acc) {
            (true, _, event, rate) => (event, rate),
            (false, acc, _, _) => {
                if self.chunk_handling == ChunkHandling::Equilibrium {
                    match self.choose_dimer_attachment_at_point(state, p, acc) {
                        (true, _, event, rate) => (event, rate),
                        (false, acc, _, _) => panic!("..."),
                    }
                } else {
                    panic!("...");
                }
            }
        },
    }
}
```

### Step 2.7: Handle `update_after_event` for PolymerAttachment

Already handled — `Event::PolymerAttachment` is matched in `update_after_event()` (line 262) and updates all surrounding points. No changes needed.

### Step 2.8: Handle `perform_event` for PolymerAttachment

Already handled — `Event::PolymerAttachment` is matched in `perform_event()` (line 524) and places tiles including duple expansion. No changes needed, as long as dimers don't involve duple tiles attaching as dimers (which would be a separate, more complex feature).

### Step 2.9: Tests

1. **Detailed balance test**: Set up a small system with `ChunkHandling::Equilibrium`. Run many steps. Verify the assembly reaches a steady state where the attachment and detachment rates are balanced (statistically).

2. **Dimer attachment rate test**: Construct a state with specific empty locations. Verify `total_dimer_attachment_rate_at_point()` returns the expected value based on known dimer concentrations and neighbor compatibility.

3. **Event selection test**: Verify `choose_dimer_attachment_at_point()` correctly selects dimers and produces `PolymerAttachment` events with the right tile pairs and positions.

4. **Depletion consistency test**: With `ChunkHandling::Equilibrium`, verify that monomer attachment rates use depleted concentrations (lower than with `ChunkHandling::None`).

5. **Regression test**: Run existing `test_ktam_dimer_detach_on` and `test_ktam_dimer_detach_off` tests — they should still pass (Phase 2 doesn't change `Detach` or `None` behavior).

6. **Round-trip test**: Python test that creates a system with `ChunkHandling::Equilibrium`, evolves it, changes `g_se`, evolves again, and verifies concentrations/rates updated properly.

---

## Phase 3: KBlock integration

### Step 3.1: Add `compute_dimer_equilibrium()` to KBlock

**File**: `rgrow-rust/src/models/kblock.rs`

Simpler than KTAM because KBlock already has physical units:

```rust
fn compute_dimer_equilibrium(&self) -> Result<equiconc::Equilibrium, GrowError> {
    let mut sys = equiconc::System::new()
        .temperature(self.temperature.to_kelvin_m());

    for (i, &conc) in self.tile_concentration.iter().enumerate() {
        if f64::from(conc) > 0.0 {
            sys = sys.monomer(&self.tile_names[i], conc.into());
        }
    }

    for ((t1, t2), e) in self.energy_ns.indexed_iter() {
        if *e > KcalPerMol::zero() {
            let delta_g: f64 = (-*e).into();
            // homodimer correction if needed
            let name = format!("{}_{}_NS", ...);
            sys = sys.complex(&name, &[...], delta_g);
        }
    }
    // WE dimers...

    sys.equilibrium().map_err(...)
}
```

### Step 3.2: Cache and apply (same pattern as KTAM)

Add `dimer_eq_concs`, `free_tile_concs` fields. Call from the KBlock equivalent of `update_system()`. Update `calc_dimers()`.

### Step 3.3: Consider replacing `fill_free_blocker_concentrations()`

KBlock's existing blocker equilibrium calculation (quadratic formula in `fill_free_blocker_concentrations()`) handles a simpler version of the same problem. equiconc could replace it and handle multi-way competition. This is optional and can be deferred — the blocker calculation is a separate concern from dimer equilibrium. But note it as a future opportunity.

### Step 3.4: Tests

Mirror the KTAM tests with KBlock-specific setup (physical temperature, kcal/mol energies).

---

## Implementation order and dependencies

```
Step 1.1  (Cargo.toml)
  |
Step 1.2  (KTAM fields)
  |
Step 1.3  (compute_dimer_equilibrium)
  |
Step 1.4  (update_system integration)
  |
Step 1.5  (calc_dimers rewrite)
  |
Step 1.6  (log dep)
  |
Step 1.7  (Phase 1 tests)
  |
  +-- Phase 1 complete, merge-worthy --
  |
Step 2.1  (effective_monomer_conc)
  |
Step 2.2  (depleted monomer attachment)
  |
Step 2.3  (dimer attachment tables)
  |
Step 2.4  (total_dimer_attachment_rate_at_point)
  |
Step 2.5  (choose_dimer_attachment_at_point)
  |
Step 2.6  (wire into event_rate / choose_event)
  |
Step 2.7-2.8  (verify existing PolymerAttachment handling works)
  |
Step 2.9  (Phase 2 tests)
  |
  +-- Phase 2 complete, merge-worthy --
  |
Steps 3.1-3.4  (KBlock, independent of Phase 2)
```

Phases 1 and 3 are independent of each other and could be done in parallel. Phase 2 depends on Phase 1.

## Risk areas

1. **Duple tiles in dimers**: Fake duple parts must be excluded as equiconc monomers, but their energy contributions are real. The `get_effective_concentration()` approach handles this, but dimer attachment of a duple tile (placing 3-4 grid cells at once) is out of scope and should be explicitly disallowed or deferred.

2. **Tile name uniqueness**: equiconc requires unique monomer and complex names. Tile names are unique in practice, but if a user provides duplicate names, the equiconc call will fail. Add a fallback to use index-based names if tile names have duplicates.

3. **Dimer attachment neighbor compatibility**: The compatibility check in `choose_dimer_attachment_at_point()` must exactly mirror what `_find_monomer_attachment_possibilities_at_point()` does for each tile individually, but applied to both tiles simultaneously. Getting this wrong would break detailed balance. Test thoroughly.

4. **Performance of dimer attachment search**: For systems with many tile types, iterating all possible dimers at every empty point could be slow. The friends-list infrastructure should be used to prune early. Profile if N_tiles > 50.

5. **equiconc convergence failures**: For extreme parameter values, equiconc might fail to converge (it returns `EquilibriumError::ConvergenceFailure`). The fallback (step 1.4) logs a warning and uses undepleted concentrations — safe but not ideal. Consider whether such failures should propagate as errors in `ChunkHandling::Equilibrium` mode (where correctness depends on the solve).
