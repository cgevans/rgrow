# Dimer Equilibrium Concentrations: Using equiconc for Accurate Calculation

## Problem Statement

rgrow's current dimer equilibrium concentration calculation assumes tile monomer concentrations are fixed — that dimerization doesn't deplete the free monomer pool. This is only valid when dimer concentrations are negligible compared to total monomer concentrations (i.e., weak binding or low concentrations). For systems with strong dimer interactions, this significantly overestimates dimer concentrations.

Additionally, dimer *attachment* (as opposed to just dimer detachment) is not implemented, despite being mentioned as a goal in the source comments (ktam.rs line 2: "the intention to eventually add dimer detachment and attachment").

The `equiconc` library at `../equiconc` solves exactly this class of problem: given total monomer concentrations and complex formation energies, compute equilibrium concentrations of all species (free monomers and complexes) with proper mass conservation.

## Current Implementation

### How dimer concentrations are calculated now

In `ktam.rs:898-937`, `calc_dimers()`:

```rust
for ((t1, t2), e) in self.energy_ns.indexed_iter() {
    if *e > 0. {
        let conc1 = self.get_effective_concentration(t1 as Tile);
        let conc2 = self.get_effective_concentration(t2 as Tile);
        let biconc: MolarSq = (conc1 * conc2).into();
        dvec.push(DimerInfo {
            t1: t1 as Tile,
            t2: t2 as Tile,
            orientation: Orientation::NS,
            formation_rate: Into::<PerMolarSecond>::into(self.kf) * biconc,
            equilibrium_conc: biconc.over_u0() * f64::exp(*e - self.alpha),
        });
    }
}
```

**Key issue**: `conc1` and `conc2` are the *total* (input) tile concentrations. The equilibrium concentration formula `[t1]*[t2] * exp(energy - alpha)` treats these as fixed, not depleted by dimerization. This is the "ideal dilute" approximation.

### Unit system in the kTAM

The kTAM uses dimensionless energies that represent `ΔG/(RT)` at an implicit, untracked temperature. These are *not* temperature-independent — they are the result of the user pre-computing the dimensionless free energies at a specific temperature of interest:
- `g_se`: dimensionless bond strength per unit glue strength (positive = favorable). Encodes `ΔG_se/(RT)` at the operating temperature.
- `alpha`: dimensionless, effectively `g_mc - ln(stoic)` (positive = favorable)
- `energy_ns[t1,t2] = g_se * glue_strength[glue]` — dimensionless
- `tile_concs[t] = stoic[t] * exp(-g_mc + alpha)` — in Molar

Because temperature is already baked into these dimensionless values, the equilibrium constants are directly:
```
K_dimer = exp(energy)    (dimensionless, 1M standard state)
[dimer] = K * [t1] * [t2] / u0
```
where `u0 = 1 M` is the standard concentration. The `exp(-alpha)` factor accounts for the fact that the dimer's "concentration equivalent" must be reduced by the same alpha factor that boosts monomer attachment.

### Unit system in KBlock

KBlock uses physical units throughout: `KcalPerMol` for energies, `Celsius`/`Kelvin` for temperature. Its `calc_dimers()` is analogous but uses `energy.times_beta(temperature)` to convert to dimensionless form. KBlock also has a precedent for equilibrium depletion calculations in `fill_free_blocker_concentrations()` (kblock.rs:410-436), which solves a quadratic for free blocker concentration given total blocker and tile glue usage — a simpler version of the same problem equiconc solves.

### What `calc_dimers()` is used for

`DimerInfo` is primarily used for:
1. **Reporting**: Showing users what dimers exist and their concentrations
2. **Dimer detachment rates**: When `ChunkHandling::Detach` is active, dimers can detach as units (ktam.rs:1806-1920)
3. **No dimer attachment**: Currently, only monomer attachment is simulated; dimers don't attach to the growing assembly

## The equiconc Library

### What it does

equiconc computes equilibrium concentrations for systems of interacting species forming multi-strand complexes. Given:
- **Monomers** with total concentrations
- **Complexes** with stoichiometric compositions and ΔG° values

It returns the equilibrium free monomer concentrations and complex concentrations, with proper mass conservation enforced.

### Algorithm

Trust-region Newton method on the convex dual problem (Dirks et al., 2007). The dual has dimension equal to the number of monomers (not the number of complexes), making it efficient even with many possible complexes. Convergence is guaranteed because the dual is strictly convex.

### API

```rust
use equiconc::System;

let eq = System::new()
    .temperature(310.15)           // K (default 37°C)
    .monomer("A", 100e-9)          // name, total_conc in M
    .monomer("B", 100e-9)
    .complex("AB", &[("A", 1), ("B", 1)], -10.0)  // name, composition, ΔG° in kcal/mol
    .equilibrium()?;

let free_a = eq.concentration("A").unwrap();    // depleted free [A]
let free_b = eq.concentration("B").unwrap();    // depleted free [B]
let ab_conc = eq.concentration("AB").unwrap();  // dimer concentration
// Mass conservation: free_a + ab_conc == 100e-9
```

### Key conventions
- **ΔG° in kcal/mol** (not kT, not kJ/mol)
- **Standard state: 1 M**
- **Concentrations in Molar**
- **Temperature in Kelvin**
- **Symmetry corrections NOT automatic** — user must include `+RT*ln(σ)` in ΔG° for homo-oligomers
- Constants: `R = 1.987204e-3 kcal/(mol·K)`

## Integration Plan

### 1. Adding equiconc as a dependency

In `rgrow-rust/Cargo.toml`:
```toml
equiconc = { path = "../equiconc" }  # development
# or
equiconc = "0.1"                      # once published
```

Note: equiconc currently builds as `cdylib` + `rlib`. Only the `rlib` target is needed for Rust dependency use. This works fine — Cargo will use the `rlib`.

### 2. Mapping kTAM energies to equiconc ΔG°

The kTAM uses dimensionless energies (e.g., `g_se`, `alpha`, `energy_ns`). equiconc uses ΔG° in kcal/mol with an explicit temperature. The key question is how to bridge these.

**The kTAM's dimensionless energies are NOT temperature-independent.** A dimensionless energy like `g_se` represents `ΔG/(RT)` at some specific temperature. Since ΔG = ΔH - TΔS, the dimensionless `g_se = (ΔH - TΔS)/(RT) = ΔH/(RT) - ΔS/R` changes with temperature. The kTAM simply doesn't track temperature — it operates at a single implicit temperature where the user has pre-computed these dimensionless values.

This means there is no physically meaningful way to "convert" the kTAM's dimensionless energies to ΔG° in kcal/mol without knowing the temperature. However, for the purpose of feeding equiconc, all we need is the correct equilibrium constant, and that is what the dimensionless energy already encodes:

```
K = exp(energy_dimensionless)    (with 1M standard state)
```

equiconc internally computes `K = exp(-ΔG° / RT)`. So we just need `ΔG°` and `T` such that `-ΔG° / RT = energy_dimensionless`. Any consistent `(ΔG°, T)` pair that satisfies this works. The simplest choice:

**Set `T = 1/R ≈ 503.22 K` so that `ΔG° = -energy_dimensionless` numerically.**

At this fictitious temperature, `exp(-ΔG°/RT) = exp(-(-E)/(R * 1/R)) = exp(E)`, which is exactly the kTAM's equilibrium constant. This is not a physical temperature — it's a mathematical convenience that makes equiconc compute the right equilibrium constants from the kTAM's pre-computed dimensionless energies.

This approach is correct because:
- The kTAM has already "baked in" the temperature when the user specified `g_se`, `g_mc`, etc.
- We only need equiconc to compute the correct mass-balance equilibrium, which depends only on K values and total concentrations
- The fictitious temperature produces exactly the right K values

**For KBlock**, which has explicit temperature (`self.temperature`) and energies in kcal/mol (`glue_links`), the mapping is direct — pass the real temperature and real ΔG° values.

### 3. Building the equiconc System from tile data

The core mapping function would look something like:

```rust
fn compute_dimer_equilibrium(&self) -> Result<equiconc::Equilibrium, GrowError> {
    // Use fictitious T = 1/R so that ΔG° = -energy_dimensionless
    // produces the correct K = exp(energy) for each dimer.
    let t_ref = 1.0 / equiconc::R;  // ~503 K

    let mut sys = equiconc::System::new().temperature(t_ref);

    // Add each tile type as a monomer
    // Total concentration = tile_concs[t] (already in M)
    for t in 1..self.tile_concs.len() {
        if self.tile_concs[t] > 0.0 {
            sys = sys.monomer(&self.tile_names[t], self.tile_concs[t]);
        }
    }

    // Add each possible dimer as a complex
    for ((t1, t2), e) in self.energy_ns.indexed_iter() {
        if *e > 0.0 && t1 > 0 && t2 > 0 {
            let delta_g = -*e;  // at T_ref = 1/R, ΔG° = -energy gives K = exp(energy)
            let name = format!("{}_{}_NS", self.tile_names[t1], self.tile_names[t2]);
            sys = sys.complex(&name, &[
                (&self.tile_names[t1], 1),
                (&self.tile_names[t2], 1),
            ], delta_g)?;
        }
    }
    // ... same for energy_we ...

    sys.equilibrium().map_err(|e| GrowError::from(e))
}
```

**Important considerations:**

- **Homodimers**: No symmetry correction is needed. The kTAM's `energy_ns[t,t]` is the bond energy for two tiles at distinguishable lattice positions (north vs south), so `K = exp(E)` and `[dimer] = [A]^2 * exp(E)` with no factor of 2. equiconc with composition `[("A", 2)]` computes `[AA] = exp(-ΔG°/RT) * [A]^2` — also no symmetry factor (equiconc documents that symmetry corrections must be manually added to ΔG°, but here none is wanted). So `ΔG° = -E` works directly for homodimers, same as heterodimers.

- **Multiple bond types per pair**: Two tiles might form both a NS dimer and a WE dimer with different energies. These are distinct complexes and should be added separately.

- **Duple tiles**: Fake duple parts should not be added as monomers. Use `get_effective_concentration()` and `should_be_counted` to filter.

- **Zero-concentration tiles**: Tile index 0 is the "empty" tile. Skip it. Also skip tiles with zero stoichiometry.

### 4. Using equilibrium results for dimer concentrations

Once the equilibrium is solved:

```rust
let eq = self.compute_dimer_equilibrium()?;

// Get depleted free monomer concentrations
for t in 1..self.tile_concs.len() {
    let free_conc = eq.concentration(&self.tile_names[t]).unwrap_or(0.0);
    // free_conc < tile_concs[t] due to depletion
}

// Get dimer concentrations
for dimer in &dimers {
    let dimer_conc = eq.concentration(&dimer.name).unwrap_or(0.0);
    // This properly accounts for depletion
}
```

### 5. Adjusting tile concentrations for depletion

Most tile assembly papers assume monomer concentrations are not depleted by dimerization — the "infinite reservoir" approximation. This is reasonable when dimer concentrations are small relative to total monomer concentrations (weak binding or low concentrations). **Depletion should therefore be optional**, not the default.

Depletion should be turned on automatically when dimer attachment is enabled (since using undepleted concentrations for both monomer *and* dimer attachment would double-count), but otherwise left off.

When enabled, the implementation stores depleted concentrations separately:

```rust
// Store free (equilibrium) concentrations separately
pub free_tile_concs: Option<Array1<Conc>>,  // None = use tile_concs (no depletion)

fn update_free_concentrations(&mut self) {
    let eq = self.compute_dimer_equilibrium().unwrap();
    let mut free = self.tile_concs.clone();
    for t in 1..self.tile_concs.len() {
        free[t] = eq.concentration(&self.tile_names[t])
            .unwrap_or(self.tile_concs[t]);
    }
    self.free_tile_concs = Some(free);
}

fn effective_monomer_conc(&self, t: usize) -> Conc {
    self.free_tile_concs.as_ref()
        .map_or(self.tile_concs[t], |fc| fc[t])
}
```

Then monomer attachment rates use `effective_monomer_conc(t)`:
```rust
// ktam.rs:1635, currently:
let rate = self.kf * self.tile_concs[t as usize];
// becomes:
let rate = self.kf * self.effective_monomer_conc(t as usize);
```

#### Depletion warning

It would be useful to warn users when dimer concentrations are large relative to monomer concentrations and depletion is *not* enabled, since this indicates the undepleted approximation is breaking down. However, computing this requires running equiconc even when depletion isn't turned on. Two options:

**Option A: Run equiconc unconditionally at system construction, warn if needed.** equiconc is fast (sub-microsecond for typical tile counts), so the cost is negligible. The warning would fire during `update_system()` or `calc_dimers()`:

```rust
// After computing equilibrium:
for t in 1..self.tile_concs.len() {
    let free = eq.concentration(&self.tile_names[t]).unwrap_or(self.tile_concs[t]);
    let depleted_fraction = 1.0 - free / self.tile_concs[t];
    if depleted_fraction > 0.1 {  // >10% depletion
        warn!("Tile {} has {:.0}% depletion from dimerization; \
               consider enabling dimer depletion", self.tile_names[t],
               depleted_fraction * 100.0);
    }
}
```

**Option B: Use the naive (undepleted) dimer concentrations as a cheap heuristic.** If `[dimer_naive] / [monomer]` exceeds a threshold, suggest enabling depletion. This avoids the equiconc dependency when it's not otherwise needed, but is less accurate (the naive formula overestimates dimers, which is exactly the problem — so this heuristic will over-warn, but at least it errs on the safe side).

**Recommendation**: Option A is cleaner. The equiconc cost is negligible, and running it always means `calc_dimers()` can return accurate equilibrium concentrations in all cases, even when depletion isn't applied to monomer attachment rates.

### 6. Implementing dimer attachment

With proper dimer concentrations, dimer attachment becomes feasible. A dimer attachment event places two tiles simultaneously at adjacent positions.

#### Rate calculation

The rate of dimer attachment at a specific pair of adjacent empty sites is:
```
rate = kf * [dimer]
```
where `[dimer]` is the equilibrium dimer concentration from equiconc, and `kf` is the same forward rate constant used for monomers.

This is the same form as monomer attachment (`kf * [monomer]`), but with the dimer concentration. The physical justification: the dimer is a single diffusing species that attaches as a unit.

#### Implementation sketch

```rust
fn find_dimer_attachment_at_point<St: State>(
    &self,
    state: &St,
    point: PointSafe2,
    mut acc: f64,
) -> (bool, f64, Event, f64) {
    // Check south: can a NS dimer attach with top tile at `point`?
    let ps = state.move_sa_s(point);
    if state.tile_at_point(PointSafe2(ps.0)) == 0 {
        for &(t1, t2, dimer_conc) in &self.ns_dimer_attachments {
            // Check compatibility with neighbors of both positions
            if self.is_compatible(state, point, t1) &&
               self.is_compatible(state, PointSafe2(ps.0), t2) {
                let rate = self.kf * dimer_conc;
                acc -= rate;
                if acc <= 0.0 {
                    return (true, acc, Event::DimerAttachment(point, t1, t2, Orientation::NS), rate);
                }
            }
        }
    }
    // ... similar for WE ...
    (false, acc, Event::None, f64::NAN)
}
```

#### New Event variant needed
```rust
pub enum Event {
    // ... existing variants ...
    DimerAttachment(PointSafe2, Tile, Tile, Orientation),
}
```

#### Rate store integration

Dimer attachment rates at each point must be added to the rate store alongside monomer attachment and detachment rates. This affects:
- `calc_n_rates_at_point()` — add dimer attachment possibilities
- `choose_event_at_point()` — handle dimer attachment selection
- `perform_event()` — place two tiles at once

The total attachment rate at a point becomes (when dimer attachment is enabled):
```
total_rate = Σ_t kf * [free_t] + Σ_d kf * [dimer_d]
```
where `[free_t]` is the depletion-adjusted monomer concentration (required when dimer attachment is active), and the sum over dimers includes all compatible dimer types that could attach with one tile at this point.

### 7. Detailed balance considerations

For detailed balance (essential for correct thermodynamics), the dimer attachment and detachment rates must satisfy:

```
rate_attach_dimer(t1,t2) / rate_detach_dimer(t1,t2) = [dimer_eq] / [empty_pair_eq]
```

The current dimer detachment rate (ktam.rs:1806-1852) is:
```rust
kf * exp(-E_t1 - E_t2 + 2*E_bond(t1,t2) + 2*alpha)
```
where `E_t1`, `E_t2` are the total bond energies of each tile to its other neighbors, and `E_bond` is the mutual bond.

With proper dimer attachment at rate `kf * [dimer]`, detailed balance requires:
```
kf * [dimer] = kf * exp(-E_t1 - E_t2 + 2*E_bond + 2*alpha)
```
This should hold at equilibrium since `[dimer] = [t1]*[t2]*exp(E_bond)/u0` and `[ti] = stoic_i * exp(-gmc + alpha)`. The `2*alpha` in the detachment rate accounts for two tiles being removed.

**With depletion-adjusted concentrations**, this balance still holds as long as the dimer attachment rate uses the equiconc-computed `[dimer]` and the detachment rate formula remains the same. The equiconc solution *is* the equilibrium point where these rates balance.

### 8. Performance considerations and recomputation

- **equiconc is fast**: The solver typically converges in 5-20 Newton iterations. For a system with N tile types, the dual problem has dimension N, and each iteration involves an N×N Cholesky solve. For typical tile systems (N < 100), this is sub-microsecond.

- **When to recompute**: The equilibrium must be recomputed whenever any parameter that affects dimer energies or monomer concentrations changes. In the kTAM, all parameter mutations already funnel through `update_system()`:
  - Python setters: `alpha`, `g_se`, `kf`, `tile_edges`, `tile_concs` (each calls `update_system()`)
  - `set_param()` trait method: handles `g_se`, `alpha`, `kf`, `tile_concs`, `glue_strengths`, `glue_links` (each calls `update_system()`)
  - `set_duples()`: calls `update_system()`
  - `from_ktam()` constructor: calls `update_system()` at the end

  Since all paths converge on `update_system()`, adding the equiconc solve there (after energy matrices are recomputed) guarantees consistency. The solve is cheap enough that rerunning it on every parameter change is fine.

  Note: `kf` changes don't affect equilibrium concentrations (they scale all rates equally), but since they already go through `update_system()`, the recomputation is harmless.

- **Memory**: Storing `free_tile_concs` doubles the concentration storage — negligible.

### 9. Interaction with existing ChunkHandling modes

The `ChunkHandling` enum already has three variants:
- `None` — no chunk events
- `Detach` — dimer detachment only
- `Equilibrium` — (not yet implemented, reserved)

The `Equilibrium` variant would enable:
1. Dimer attachment using equiconc-computed concentrations
2. Monomer concentration depletion (required for consistency when dimers attach)
3. Dimer detachment (as in `Detach` mode)

Depletion should be **automatically enabled** when `ChunkHandling::Equilibrium` is active, since using undepleted monomer concentrations alongside dimer attachment would double-count material. For `ChunkHandling::Detach` and `None`, depletion remains off by default (matching standard kTAM assumptions), though a separate flag could allow opting in.

A possible phased approach:
- **Phase 1**: Use equiconc in `calc_dimers()` for accurate reporting and depletion warnings; no simulation behavior changes
- **Phase 2**: Implement dimer attachment + automatic depletion under `ChunkHandling::Equilibrium`
- **Phase 3** (optional): Add a standalone depletion flag for use with `Detach` or `None` modes

### 10. KBlock integration

KBlock already has physical temperature and energies in kcal/mol, making integration more direct:

```rust
fn compute_dimer_equilibrium(&self) -> Result<equiconc::Equilibrium, ...> {
    let mut sys = equiconc::System::new()
        .temperature(self.temperature.to_kelvin_m());

    for (i, &conc) in self.tile_concentration.iter().enumerate() {
        if conc > Molar::zero() {
            sys = sys.monomer(&self.tile_names[i], conc.into());
        }
    }

    // Add dimers from energy_ns and energy_we
    // ΔG° is already in kcal/mol (from glue_links)
    for ((t1, t2), e) in self.energy_ns.indexed_iter() {
        if *e > KcalPerMol::zero() {
            let delta_g: f64 = (-*e).into();  // negate: favorable binding = negative ΔG°
            sys = sys.complex(..., delta_g);
        }
    }

    sys.equilibrium()
}
```

KBlock also has a precedent in `fill_free_blocker_concentrations()` (kblock.rs:410-436) which solves for free blocker concentration with depletion via a quadratic formula. equiconc would replace this quadratic with a general multi-species solver — and could handle cases where blockers compete with each other, which the quadratic cannot.

### 11. Considerations for larger complexes

equiconc can handle complexes larger than dimers. While rgrow currently only considers dimers for chunk handling, the framework could be extended:

- **Trimers, tetramers**: If `ChunkSize` is extended, equiconc naturally handles these — just add complexes with stoichiometry counts > 1 or compositions of 3+ distinct monomers.
- **Competing complexes**: equiconc correctly handles competition between complexes for shared monomers. For example, if tile A can form dimers with B *and* with C, the equilibrium properly accounts for A being shared between both complexes.

### 12. Summary of required changes

| Component | Change | Difficulty |
|-----------|--------|------------|
| `Cargo.toml` | Add equiconc dependency | Trivial |
| `KTAM` struct | Add optional `free_tile_concs` field | Low |
| `KTAM::update_system()` | Call equiconc after computing energies; warn on high depletion | Medium |
| `KTAM::calc_dimers()` | Use equiconc results instead of naive formula | Medium |
| `KTAM::_find_monomer_attachment_possibilities_at_point()` | Use `free_tile_concs` when depletion enabled | Low |
| `Event` enum | Add `DimerAttachment` variant | Low |
| `KTAM` | New `find_dimer_attachment_at_point()` method | High |
| `KTAM::calc_n_rates_at_point()` | Include dimer attachment rates | High |
| `KTAM::choose_event_at_point()` | Handle dimer attachment selection | High |
| `KTAM::perform_event()` | Place two tiles for dimer attachment | Medium |
| `KBlock` | Analogous changes, simpler due to physical units | Medium |
| Python bindings | Expose new fields/methods | Low |
| Tests | New tests for depletion and dimer attachment | Medium |

### 13. Open questions

1. **What about larger assemblies depleting monomers?** The equiconc approach handles solution-phase equilibrium among monomers and small complexes. Monomers incorporated into the growing crystal also deplete the pool, but this is a dynamic effect not captured by a one-time equilibrium calculation. This is a separate problem (reservoir depletion) already discussed in tile assembly literature.

2. **Rate constant for dimer attachment**: Should `kf` be the same for monomers and dimers? Physically, dimers diffuse more slowly (by ~√2 for a dimer vs monomer), so `kf_dimer ≈ kf / √2`. But the standard kTAM convention is to use a single `kf`. This could be a configurable parameter.

3. **Interaction with duples**: Duple tiles (which are conceptually single tiles occupying two grid positions) already handle two-position placement. The dimer attachment mechanism is conceptually similar but for genuinely distinct tiles. The implementation should reuse duple placement machinery where possible.
