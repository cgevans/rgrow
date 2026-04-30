//! SDC2D — scaffolded DNA tile assembly in 2D.
//!
//! Strands have five glues: West, North, East, South, and Bottom (scaffold-
//! facing). The scaffold is a 2D `Array2<Glue>` of shape `(nrows, ncols)`;
//! one state simulates one scaffold. All four lateral edges share a single
//! glue namespace. Energy uses the unitful (kcal/mol, Molar, Kelvin)
//! convention with per-glue `(ΔG_37, ΔS)` parameters, mirroring SDC1D.

use std::collections::HashMap;
use std::sync::OnceLock;

use ndarray::prelude::{Array1, Array2};
use num_traits::Zero;
use serde::{Deserialize, Serialize};

use crate::base::{Glue, GrowError, Tile};
use crate::canvas::{PointSafe2, PointSafeHere};
use crate::colors::get_color_or_random;
use crate::state::State;
use crate::system::{Event, NeededUpdate, System, TileBondInfo};
use crate::units::*;

use super::sdc_common::{get_or_generate, gsorseq_to_gs, self_and_inverse, GsOrSeq, RefOrPair};

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub(crate) const WEST_GLUE_INDEX: usize = 0;
pub(crate) const NORTH_GLUE_INDEX: usize = 1;
pub(crate) const EAST_GLUE_INDEX: usize = 2;
pub(crate) const SOUTH_GLUE_INDEX: usize = 3;
pub(crate) const BOTTOM_GLUE_INDEX: usize = 4;
pub(crate) const N_GLUES_PER_STRAND: usize = 5;

#[cfg_attr(feature = "python", pyclass(subclass, module = "rgrow.rgrow"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDC2D {
    pub strand_names: Vec<String>,
    pub glue_names: Vec<String>,
    pub colors: Vec<[u8; 4]>,

    /// 2D scaffold of glue IDs, shape `(nrows, ncols)`.
    pub scaffold: Array2<Glue>,
    pub scaffold_concentration: Molar,

    /// `(n_strands, 5)` glue table: rows = strand, cols indexed by
    /// `WEST/NORTH/EAST/SOUTH/BOTTOM_GLUE_INDEX`.
    pub strand_glues: Array2<Glue>,
    pub strand_concentration: Array1<Molar>,

    /// ΔG at 37°C, indexed `[(glue_a, glue_b)]`. Single namespace shared by
    /// all four lateral edges and by the bottom/scaffold edge.
    pub delta_g_matrix: Array2<KcalPerMol>,
    /// ΔS, indexed `[(glue_a, glue_b)]`.
    pub entropy_matrix: Array2<KcalPerMolKelvin>,

    pub kf: PerMolarSecond,
    /// Kept private so callers go through `change_temperature_to`, which
    /// invalidates the lazy energy caches.
    temperature: Kelvin,

    /// Per-position list of strand IDs that can bind to the scaffold here,
    /// shape `(nrows, ncols)`.
    pub friends_btm: Array2<Vec<Tile>>,

    /// Pinned strands: a strand at one of these points cannot detach.
    pub seed: HashMap<PointSafe2, Tile>,

    /// Lazy β·ΔG cache for west↔east strand pairs, shape `(n, n)`.
    /// Index `[west, east]` = west strand's east glue × east strand's west glue.
    #[serde(skip)]
    strand_we_energy_bonds: Array2<OnceLock<f64>>,
    /// Lazy β·ΔG cache for north↔south strand pairs, shape `(n, n)`.
    /// Index `[north, south]` = north strand's south glue × south strand's north glue.
    #[serde(skip)]
    strand_ns_energy_bonds: Array2<OnceLock<f64>>,
    /// Lazy β·ΔG cache for strand↔scaffold bonds, shape `(nrows*ncols, n_strands)`.
    /// Key is `(row * ncols + col, strand)`.
    #[serde(skip)]
    scaffold_energy_bonds: Array2<OnceLock<f64>>,
}

impl SDC2D {
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.scaffold.nrows()
    }

    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.scaffold.ncols()
    }

    #[inline(always)]
    pub fn n_strands(&self) -> usize {
        self.strand_names.len()
    }

    #[inline(always)]
    pub fn temperature(&self) -> Kelvin {
        self.temperature
    }

    #[inline(always)]
    fn flat_scaffold_index(&self, row: usize, col: usize) -> usize {
        row * self.ncols() + col
    }

    /// ΔG(T) = ΔG_37 - (T - 37°C) · ΔS.
    pub fn glue_glue_dg(&self, a: Glue, b: Glue) -> KcalPerMol {
        self.delta_g_matrix[(a, b)]
            - (self.temperature - Celsius(37.0)) * self.entropy_matrix[(a, b)]
    }

    /// β·ΔG for a west-east strand pair (`west` to the west of `east`).
    pub fn bond_we(&self, west: Tile, east: Tile) -> f64 {
        *self.strand_we_energy_bonds[(west as usize, east as usize)].get_or_init(|| {
            let g_w = self.strand_glues[(west as usize, EAST_GLUE_INDEX)];
            let g_e = self.strand_glues[(east as usize, WEST_GLUE_INDEX)];
            self.glue_glue_dg(g_w, g_e).times_beta(self.temperature)
        })
    }

    /// β·ΔG for a north-south strand pair (`north` to the north of `south`).
    pub fn bond_ns(&self, north: Tile, south: Tile) -> f64 {
        *self.strand_ns_energy_bonds[(north as usize, south as usize)].get_or_init(|| {
            let g_n = self.strand_glues[(north as usize, SOUTH_GLUE_INDEX)];
            let g_s = self.strand_glues[(south as usize, NORTH_GLUE_INDEX)];
            self.glue_glue_dg(g_n, g_s).times_beta(self.temperature)
        })
    }

    /// β·ΔG for a strand bound to the scaffold at `(row, col)`.
    pub fn bond_with_scaffold(&self, row: usize, col: usize, strand: Tile) -> f64 {
        let key = self.flat_scaffold_index(row, col);
        *self.scaffold_energy_bonds[(key, strand as usize)].get_or_init(|| {
            let scaffold_glue = self.scaffold[(row, col)];
            let strand_glue = self.strand_glues[(strand as usize, BOTTOM_GLUE_INDEX)];
            self.glue_glue_dg(scaffold_glue, strand_glue)
                .times_beta(self.temperature)
        })
    }

    pub fn is_seed(&self, p: &PointSafe2) -> bool {
        self.seed.contains_key(p)
    }

    /// Clear the lazy energy caches and rebuild `friends_btm`. Call after
    /// any change that affects glue energies (temperature, concentrations,
    /// glue table).
    pub fn update_system(&mut self) {
        self.empty_cache();
        self.generate_friends_btm();
    }

    fn empty_cache(&mut self) {
        let n = self.n_strands();
        let scaff = self.nrows() * self.ncols();
        self.strand_we_energy_bonds = Array2::default((n, n));
        self.strand_ns_energy_bonds = Array2::default((n, n));
        self.scaffold_energy_bonds = Array2::default((scaff, n));
    }

    fn generate_friends_btm(&mut self) {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let mut friends: Array2<Vec<Tile>> = Array2::from_elem((nrows, ncols), Vec::new());
        for r in 0..nrows {
            for c in 0..ncols {
                let scaffold_glue = self.scaffold[(r, c)];
                for (strand_idx, &strand_glue) in self
                    .strand_glues
                    .index_axis(ndarray::Axis(1), BOTTOM_GLUE_INDEX)
                    .iter()
                    .enumerate()
                {
                    if strand_idx == 0 {
                        continue;
                    }
                    if self.delta_g_matrix[(scaffold_glue, strand_glue)] != KcalPerMol::zero()
                        || self.entropy_matrix[(scaffold_glue, strand_glue)]
                            != KcalPerMolKelvin::zero()
                    {
                        friends[(r, c)].push(strand_idx as Tile);
                    }
                }
            }
        }
        self.friends_btm = friends;
    }

    /// Set temperature (Celsius or Kelvin) and invalidate caches.
    pub fn change_temperature_to(&mut self, temperature: impl Into<Kelvin>) {
        self.temperature = temperature.into();
        self.update_system();
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub struct SDC2DStrand {
    pub name: Option<String>,
    pub color: Option<String>,
    /// Strand concentration in Molar.
    pub concentration: f64,
    pub west_glue: Option<String>,
    pub north_glue: Option<String>,
    pub east_glue: Option<String>,
    pub south_glue: Option<String>,
    pub bottom_glue: Option<String>,
}

#[derive(Debug)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub struct SDC2DParams {
    pub strands: Vec<SDC2DStrand>,
    /// `[row][col]` of glue names. `None` = no scaffold glue (binds nothing).
    /// All rows must have the same length.
    pub scaffold: Vec<Vec<Option<String>>>,
    /// Scaffold concentration in Molar.
    pub scaffold_concentration: f64,
    /// (ΔG_37, ΔS) or DNA sequence per glue or glue-pair.
    pub glue_dg_s: HashMap<RefOrPair, GsOrSeq>,
    /// Forward rate constant in 1/(M·s).
    pub k_f: f64,
    /// Temperature in Celsius.
    pub temperature: f64,
    /// Anchor strands: each `(row, col, strand_name)` pins the named strand
    /// to that scaffold position. The strand cannot detach.
    pub seed: Vec<(usize, usize, String)>,
}

impl SDC2D {
    pub fn from_params(params: SDC2DParams) -> Self {
        let SDC2DParams {
            strands,
            scaffold,
            scaffold_concentration,
            glue_dg_s,
            k_f,
            temperature,
            seed,
        } = params;

        // Validate scaffold rectangularity.
        let nrows = scaffold.len();
        assert!(nrows > 0, "SDC2D scaffold must have at least one row");
        let ncols = scaffold[0].len();
        assert!(
            ncols > 0,
            "SDC2D scaffold rows must have at least one column"
        );
        for (i, row) in scaffold.iter().enumerate() {
            assert_eq!(
                row.len(),
                ncols,
                "SDC2D scaffold row {i} has length {} but row 0 has length {ncols}",
                row.len()
            );
        }

        // Strand 0 is the null strand.
        let n_strands = strands.len() + 1;
        let mut strand_names: Vec<String> = Vec::with_capacity(n_strands);
        let mut strand_colors: Vec<[u8; 4]> = Vec::with_capacity(n_strands);
        let mut strand_concs = Array1::<f64>::zeros(n_strands);
        strand_names.push("null".to_string());
        strand_colors.push([0, 0, 0, 0]);

        let mut glue_name_map: HashMap<String, usize> = HashMap::new();
        let mut gluenum: usize = 1;
        let mut strand_glues = Array2::<Glue>::zeros((n_strands, N_GLUES_PER_STRAND));

        for (i, strand) in strands.into_iter().enumerate() {
            let idx = i + 1;
            strand_names.push(strand.name.clone().unwrap_or_else(|| i.to_string()));
            let color = get_color_or_random(strand.color.as_deref()).unwrap();
            strand_colors.push(color);
            strand_concs[idx] = strand.concentration;

            strand_glues[(idx, WEST_GLUE_INDEX)] =
                get_or_generate(&mut glue_name_map, &mut gluenum, strand.west_glue);
            strand_glues[(idx, NORTH_GLUE_INDEX)] =
                get_or_generate(&mut glue_name_map, &mut gluenum, strand.north_glue);
            strand_glues[(idx, EAST_GLUE_INDEX)] =
                get_or_generate(&mut glue_name_map, &mut gluenum, strand.east_glue);
            strand_glues[(idx, SOUTH_GLUE_INDEX)] =
                get_or_generate(&mut glue_name_map, &mut gluenum, strand.south_glue);
            strand_glues[(idx, BOTTOM_GLUE_INDEX)] =
                get_or_generate(&mut glue_name_map, &mut gluenum, strand.bottom_glue);
        }

        // Pre-register every scaffold glue (so the energy matrix sizing covers them).
        for row in scaffold.iter() {
            for cell in row.iter() {
                let _ = get_or_generate(&mut glue_name_map, &mut gluenum, cell.clone());
            }
        }

        let mut delta_g_matrix = Array2::<KcalPerMol>::zeros((gluenum, gluenum));
        let mut entropy_matrix = Array2::<KcalPerMolKelvin>::zeros((gluenum, gluenum));

        for (key, gs_or_seq) in glue_dg_s.iter() {
            let (dg, ds) = gsorseq_to_gs(gs_or_seq);
            let (i_name, j_name) = match key {
                RefOrPair::Ref(r) => {
                    let (_, base, inverse) = self_and_inverse(r);
                    (base, inverse)
                }
                RefOrPair::Pair(r1, r2) => {
                    let (r1_is_from, r1f, r1t) = self_and_inverse(r1);
                    let (r2_is_from, r2f, r2t) = self_and_inverse(r2);
                    (
                        if r1_is_from { r1f } else { r1t },
                        if r2_is_from { r2f } else { r2t },
                    )
                }
            };
            let (i, j) = match (glue_name_map.get(&i_name), glue_name_map.get(&j_name)) {
                (Some(&x), Some(&y)) => (x, y),
                _ => continue,
            };
            delta_g_matrix[(i, j)] = dg;
            delta_g_matrix[(j, i)] = dg;
            entropy_matrix[(i, j)] = ds;
            entropy_matrix[(j, i)] = ds;
        }

        // Build the Array2<Glue> scaffold from the row-major Vec<Vec<...>>.
        let mut scaffold_glues = Array2::<Glue>::zeros((nrows, ncols));
        for (r, row) in scaffold.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if let Some(name) = cell {
                    scaffold_glues[(r, c)] = *glue_name_map
                        .get(name)
                        .unwrap_or_else(|| panic!("Scaffold glue '{name}' not found"));
                }
            }
        }

        let mut glue_names = vec![String::new(); gluenum];
        for (s, i) in glue_name_map.iter() {
            glue_names[*i] = s.clone();
        }

        // Resolve seed strand-name strings to Tile IDs.
        let mut seed_map: HashMap<PointSafe2, Tile> = HashMap::with_capacity(seed.len());
        for (r, c, name) in seed {
            assert!(
                r < nrows && c < ncols,
                "Seed position ({r}, {c}) out of bounds for {nrows}x{ncols} scaffold"
            );
            let tile = strand_names
                .iter()
                .position(|n| n == &name)
                .unwrap_or_else(|| panic!("Seed strand '{name}' not found"))
                as Tile;
            seed_map.insert(PointSafe2((r, c)), tile);
        }

        let strand_concentration = strand_concs.mapv(Molar::new);
        let kf = PerMolarSecond::new(k_f);
        let temperature: Kelvin = Celsius(temperature).into();
        let scaffold_count = nrows * ncols;
        let mut s = SDC2D {
            strand_names,
            glue_names,
            colors: strand_colors,
            scaffold: scaffold_glues,
            scaffold_concentration: Molar::new(scaffold_concentration),
            strand_glues,
            strand_concentration,
            delta_g_matrix,
            entropy_matrix,
            kf,
            temperature,
            friends_btm: Array2::from_elem((nrows, ncols), Vec::new()),
            seed: seed_map,
            strand_we_energy_bonds: Array2::default((n_strands, n_strands)),
            strand_ns_energy_bonds: Array2::default((n_strands, n_strands)),
            scaffold_energy_bonds: Array2::default((scaffold_count, n_strands)),
        };
        s.update_system();
        s
    }
}

// ─── Hot path ────────────────────────────────────────────────────────────────

impl SDC2D {
    /// Sum of β·ΔG over scaffold + W + E + N + S bonds for a strand at `p`.
    /// Out-of-bounds neighbors and empty (tile == 0) neighbors contribute 0.
    fn bond_energy_of_strand<S: State>(&self, state: &S, p: PointSafe2, strand: Tile) -> f64 {
        let (row, col) = p.0;
        let mut e = self.bond_with_scaffold(row, col, strand);

        let pw = state.move_sa_w(p);
        if state.inbounds(pw.0) {
            let n = state.v_sh(pw);
            if n != 0 {
                e += self.bond_we(n, strand);
            }
        }
        let pe = state.move_sa_e(p);
        if state.inbounds(pe.0) {
            let n = state.v_sh(pe);
            if n != 0 {
                e += self.bond_we(strand, n);
            }
        }
        let pn = state.move_sa_n(p);
        if state.inbounds(pn.0) {
            let n = state.v_sh(pn);
            if n != 0 {
                e += self.bond_ns(n, strand);
            }
        }
        let ps = state.move_sa_s(p);
        if state.inbounds(ps.0) {
            let n = state.v_sh(ps);
            if n != 0 {
                e += self.bond_ns(strand, n);
            }
        }
        e
    }

    pub fn monomer_detachment_rate_at_point<S: State>(
        &self,
        state: &S,
        p: PointSafe2,
    ) -> PerSecond {
        let strand = state.tile_at_point(p);
        if strand == 0 {
            return PerSecond::zero();
        }
        if self.is_seed(&p) {
            return PerSecond::zero();
        }
        let bond_energy = self.bond_energy_of_strand(state, p, strand);
        self.kf * Molar::u0_times(bond_energy.exp())
    }

    pub fn total_monomer_attachment_rate_at_point<S: State>(
        &self,
        _state: &S,
        p: PointSafe2,
    ) -> PerSecond {
        let (row, col) = p.0;
        let mut total = PerSecond::zero();
        for &t in self.friends_btm[(row, col)].iter() {
            total += self.kf * self.strand_concentration[t as usize];
        }
        total
    }

    fn choose_monomer_attachment_at_point<S: State>(
        &self,
        _state: &S,
        p: PointSafe2,
        mut acc: PerSecond,
    ) -> (bool, PerSecond, Event, f64) {
        let (row, col) = p.0;
        for &strand in self.friends_btm[(row, col)].iter() {
            let rate = self.kf * self.strand_concentration[strand as usize];
            acc -= rate;
            if acc <= PerSecond::zero() {
                return (true, acc, Event::MonomerAttachment(p, strand), rate.into());
            }
        }
        (false, acc, Event::None, f64::NAN)
    }

    fn choose_monomer_detachment_at_point<S: State>(
        &self,
        state: &S,
        p: PointSafe2,
        mut acc: PerSecond,
    ) -> (bool, PerSecond, Event, f64) {
        let rate = self.monomer_detachment_rate_at_point(state, p);
        acc -= rate;
        if acc > PerSecond::zero() {
            return (false, acc, Event::None, rate.into());
        }
        (true, acc, Event::MonomerDetachment(p), rate.into())
    }

    fn update_monomer_point<S: State>(&self, state: &mut S, p: &PointSafe2) {
        let mut points: Vec<(PointSafeHere, PerSecond)> = Vec::with_capacity(5);
        let pw = state.move_sa_w(*p);
        if state.inbounds(pw.0) {
            points.push((pw, self.event_rate_at_point(state, pw)));
        }
        let pe = state.move_sa_e(*p);
        if state.inbounds(pe.0) {
            points.push((pe, self.event_rate_at_point(state, pe)));
        }
        let pn = state.move_sa_n(*p);
        if state.inbounds(pn.0) {
            points.push((pn, self.event_rate_at_point(state, pn)));
        }
        let ps = state.move_sa_s(*p);
        if state.inbounds(ps.0) {
            points.push((ps, self.event_rate_at_point(state, ps)));
        }
        let ph = PointSafeHere(p.0);
        points.push((ph, self.event_rate_at_point(state, ph)));
        state.update_multiple(&points);
    }
}

impl System for SDC2D {
    fn update_after_event<S: State>(&self, state: &mut S, event: &Event) {
        match event {
            Event::MonomerAttachment(p, _)
            | Event::MonomerDetachment(p)
            | Event::MonomerChange(p, _) => self.update_monomer_point(state, p),
            _ => panic!("Event type not supported in SDC2D: {event:?}"),
        }
    }

    fn perform_event<S: State>(&self, state: &mut S, event: &Event) -> f64 {
        match event {
            Event::None => panic!("Being asked to perform null event."),
            Event::MonomerAttachment(p, strand) => {
                state.update_attachment(*strand);
                state.set_sa(p, strand);
            }
            Event::MonomerDetachment(p) => {
                let strand = state.tile_at_point(*p);
                state.update_detachment(strand);
                state.set_sa(p, &0);
            }
            Event::MonomerChange(p, strand) => state.set_sa(p, strand),
            _ => panic!("Event type not supported in SDC2D: {event:?}"),
        };
        f64::NAN
    }

    fn event_rate_at_point<S: State>(&self, state: &S, p: PointSafeHere) -> PerSecond {
        if !state.inbounds(p.0) {
            return PerSecond::zero();
        }
        let pp = PointSafe2(p.0);
        match state.tile_at_point(pp) {
            0 => self.total_monomer_attachment_rate_at_point(state, pp),
            _ => self.monomer_detachment_rate_at_point(state, pp),
        }
    }

    fn choose_event_at_point<S: State>(
        &self,
        state: &S,
        p: PointSafe2,
        acc: PerSecond,
    ) -> (Event, f64) {
        let (occur, acc, event, rate) = self.choose_monomer_detachment_at_point(state, p, acc);
        if occur {
            return (event, rate);
        }
        let (occur, _acc, event, rate) = self.choose_monomer_attachment_at_point(state, p, acc);
        if occur {
            return (event, rate);
        }
        panic!(
            "SDC2D: no event chosen at {p:?} with accumulator residual {_acc:?} (state may be stale)"
        );
    }

    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)> {
        self.seed.iter().map(|(&p, &t)| (p, t)).collect()
    }

    fn calc_mismatch_locations<S: State>(&self, state: &S) -> Array2<usize> {
        let threshold = -0.1;
        let mut out = Array2::<usize>::zeros((state.nrows(), state.ncols()));
        for i in 0..state.nrows() {
            for j in 0..state.ncols() {
                if !state.inbounds((i, j)) {
                    continue;
                }
                let p = PointSafe2((i, j));
                let t = state.tile_at_point(p);
                if t == 0 {
                    continue;
                }
                let te = state.tile_to_e(p);
                let tw = state.tile_to_w(p);
                let tn = state.tile_to_n(p);
                let ts = state.tile_to_s(p);
                let mm_e = ((te != 0) & (self.bond_we(t, te) > threshold)) as usize;
                let mm_w = ((tw != 0) & (self.bond_we(tw, t) > threshold)) as usize;
                let mm_n = ((tn != 0) & (self.bond_ns(tn, t) > threshold)) as usize;
                let mm_s = ((ts != 0) & (self.bond_ns(t, ts) > threshold)) as usize;
                out[(i, j)] = 8 * mm_n + 4 * mm_e + 2 * mm_s + mm_w;
            }
        }
        out
    }

    fn set_param(
        &mut self,
        name: &str,
        value: Box<dyn std::any::Any>,
    ) -> Result<NeededUpdate, GrowError> {
        match name {
            "kf" => {
                let kf = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.kf = PerMolarSecond::from(*kf);
                self.update_system();
                Ok(NeededUpdate::NonZero)
            }
            "strand_concentrations" => {
                let concs = value
                    .downcast_ref::<Array1<Molar>>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.strand_concentration.clone_from(concs);
                self.update_system();
                Ok(NeededUpdate::NonZero)
            }
            "temperature" => {
                let t = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.change_temperature_to(Celsius(*t));
                Ok(NeededUpdate::NonZero)
            }
            _ => Err(GrowError::NoParameter(name.to_string())),
        }
    }

    fn get_param(&self, name: &str) -> Result<Box<dyn std::any::Any>, GrowError> {
        match name {
            "kf" => Ok(Box::new(f64::from(self.kf))),
            "strand_concentrations" => Ok(Box::new(self.strand_concentration.clone())),
            "temperature" => Ok(Box::new(self.temperature.to_celsius().0)),
            _ => Err(GrowError::NoParameter(name.to_string())),
        }
    }

    fn list_parameters(&self) -> Vec<crate::system::ParameterInfo> {
        use crate::system::ParameterInfo;
        vec![
            ParameterInfo {
                name: "temperature".to_string(),
                units: "°C".to_string(),
                default_increment: 1.0,
                min_value: Some(0.0),
                max_value: Some(100.0),
                description: Some("Simulation temperature".to_string()),
                current_value: self.temperature.to_celsius().0,
            },
            ParameterInfo {
                name: "kf".to_string(),
                units: "M/s".to_string(),
                default_increment: 1e5,
                min_value: Some(0.0),
                max_value: None,
                description: Some("Forward reaction rate constant".to_string()),
                current_value: f64::from(self.kf),
            },
        ]
    }

    fn system_info(&self) -> String {
        format!(
            "SDC2D with {}x{} scaffold and {} strands",
            self.nrows(),
            self.ncols(),
            self.n_strands(),
        )
    }
}

impl TileBondInfo for SDC2D {
    fn tile_colors(&self) -> &Vec<[u8; 4]> {
        &self.colors
    }
    fn tile_names(&self) -> &[String] {
        &self.strand_names
    }
    fn bond_names(&self) -> &[String] {
        &self.glue_names
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl SDC2D {
    #[new]
    fn py_new(params: SDC2DParams) -> Self {
        SDC2D::from_params(params)
    }

    #[getter(kf)]
    fn py_get_kf(&self) -> f64 {
        f64::from(self.kf)
    }

    #[setter(kf)]
    fn py_set_kf(&mut self, kf: f64) {
        self.kf = PerMolarSecond::from(kf);
        self.update_system();
    }

    #[getter(temperature)]
    fn py_get_temperature(&self) -> f64 {
        self.temperature.to_celsius().0
    }

    #[setter(temperature)]
    fn py_set_temperature(&mut self, temperature_c: f64) {
        self.change_temperature_to(Celsius(temperature_c));
    }

    #[pyo3(name = "nrows")]
    fn py_nrows(&self) -> usize {
        self.nrows()
    }

    #[pyo3(name = "ncols")]
    fn py_ncols(&self) -> usize {
        self.ncols()
    }

    #[pyo3(name = "n_strands")]
    fn py_n_strands(&self) -> usize {
        self.n_strands()
    }

    #[getter(strand_names)]
    fn py_strand_names(&self) -> Vec<String> {
        self.strand_names.clone()
    }

    #[getter(glue_names)]
    fn py_glue_names(&self) -> Vec<String> {
        self.glue_names.clone()
    }

    #[pyo3(name = "scaffold_glue_at")]
    fn py_scaffold_glue_at(&self, row: usize, col: usize) -> Option<String> {
        let g = self.scaffold[(row, col)];
        if g == 0 {
            None
        } else {
            Some(self.glue_names[g].clone())
        }
    }

    #[pyo3(name = "friends_at")]
    fn py_friends_at(&self, row: usize, col: usize) -> Vec<u32> {
        self.friends_btm[(row, col)].clone()
    }

    #[pyo3(name = "strand_concentrations")]
    fn py_strand_concentrations(&self) -> Vec<f64> {
        self.strand_concentration
            .iter()
            .map(|m| f64::from(*m))
            .collect()
    }

    #[pyo3(name = "mfe_config")]
    fn py_mfe_config(&self) -> (Vec<Vec<Tile>>, f64) {
        self.mfe_configuration()
    }

    #[pyo3(name = "state_g")]
    fn py_state_g(&self, state: Vec<Vec<u32>>) -> f64 {
        self.state_g(&state)
    }

    #[pyo3(name = "log_partition_function")]
    fn py_log_partition_function(&self) -> f64 {
        self.log_partition_function()
    }

    #[pyo3(name = "partition_function")]
    fn py_partition_function(&self) -> f64 {
        self.partition_function()
    }

    #[pyo3(name = "log_partial_partition_function")]
    fn py_log_partial_partition_function(&self, constraints: Vec<Vec<Vec<u32>>>) -> f64 {
        self.log_partial_partition_function(constraints)
    }

    #[pyo3(name = "partial_partition_function")]
    fn py_partial_partition_function(&self, constraints: Vec<Vec<Vec<u32>>>) -> f64 {
        self.partial_partition_function(constraints)
    }

    #[pyo3(name = "probability_of_state")]
    fn py_probability_of_state(&self, state: Vec<Vec<u32>>) -> f64 {
        self.probability_of_state(&state)
    }

    #[pyo3(name = "probability_of_constrained_configurations")]
    fn py_probability_of_constrained_configurations(&self, constraints: Vec<Vec<Vec<u32>>>) -> f64 {
        self.probability_of_constrained_configurations(constraints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a trivial 2x2 system: one strand "A" with bottom glue "g",
    /// scaffold of "g*" everywhere, and a self-binding `gΔG = -5, ΔS = 0`.
    fn minimal_2x2_params() -> SDC2DParams {
        let mut glue_dg_s = HashMap::new();
        glue_dg_s.insert(RefOrPair::Ref("g".into()), GsOrSeq::GS((-5.0, 0.0)));
        SDC2DParams {
            strands: vec![SDC2DStrand {
                name: Some("A".into()),
                color: Some("red".into()),
                concentration: 1e-6,
                west_glue: None,
                north_glue: None,
                east_glue: None,
                south_glue: None,
                bottom_glue: Some("g".into()),
            }],
            scaffold: vec![
                vec![Some("g*".into()), Some("g*".into())],
                vec![Some("g*".into()), Some("g*".into())],
            ],
            scaffold_concentration: 1e-9,
            glue_dg_s,
            k_f: 1e6,
            temperature: 37.0,
            seed: vec![],
        }
    }

    #[test]
    fn test_from_params_roundtrip() {
        let sys = SDC2D::from_params(minimal_2x2_params());
        assert_eq!(sys.nrows(), 2);
        assert_eq!(sys.ncols(), 2);
        assert_eq!(sys.n_strands(), 2);
        assert_eq!(sys.strand_glues.shape(), &[2, 5]);
        assert_eq!(sys.scaffold.shape(), &[2, 2]);
        assert_eq!(sys.friends_btm.shape(), &[2, 2]);
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(sys.friends_btm[(r, c)], vec![1u32]);
            }
        }
    }

    #[test]
    fn test_glue_glue_dg_temperature_dependence() {
        // Set up two glues a/a* with ΔG_37 = -2, ΔS = -0.01 kcal/mol/K.
        let mut glue_dg_s = HashMap::new();
        glue_dg_s.insert(RefOrPair::Ref("a".into()), GsOrSeq::GS((-2.0, -0.01)));
        let params = SDC2DParams {
            strands: vec![SDC2DStrand {
                name: Some("X".into()),
                color: None,
                concentration: 1e-6,
                west_glue: None,
                north_glue: None,
                east_glue: None,
                south_glue: None,
                bottom_glue: Some("a".into()),
            }],
            scaffold: vec![vec![Some("a*".into())]],
            scaffold_concentration: 1e-9,
            glue_dg_s,
            k_f: 1e6,
            temperature: 37.0,
            seed: vec![],
        };
        let mut sys = SDC2D::from_params(params);
        let a = sys.glue_names.iter().position(|s| s == "a").unwrap();
        let astar = sys.glue_names.iter().position(|s| s == "a*").unwrap();

        // At 37C: ΔG = -2 - (310.15 - 310.15) * (-0.01) = -2.
        let dg_37 = sys.glue_glue_dg(a, astar);
        assert!((f64::from(dg_37) - (-2.0)).abs() < 1e-12);

        // Bump T by 10 K: ΔG = -2 - 10 * (-0.01) = -1.9.
        sys.change_temperature_to(Celsius(47.0));
        let dg_47 = sys.glue_glue_dg(a, astar);
        assert!((f64::from(dg_47) - (-1.9)).abs() < 1e-9);
    }

    #[test]
    fn test_bond_caches_lazy_init() {
        let sys = SDC2D::from_params(minimal_2x2_params());
        let v1 = sys.bond_with_scaffold(0, 0, 1);
        let v2 = sys.bond_with_scaffold(0, 0, 1);
        assert_eq!(v1, v2);
        // Different positions should hit different cache slots; same value here
        // because the scaffold is uniform.
        let v3 = sys.bond_with_scaffold(1, 1, 1);
        assert_eq!(v1, v3);

        let v_we1 = sys.bond_we(1, 1);
        let v_we2 = sys.bond_we(1, 1);
        assert_eq!(v_we1, v_we2);

        let v_ns1 = sys.bond_ns(1, 1);
        let v_ns2 = sys.bond_ns(1, 1);
        assert_eq!(v_ns1, v_ns2);
    }

    #[test]
    fn test_friends_btm_per_position() {
        // Two strands and a scaffold where the two positions match different
        // bottom glues. Verify per-position friends differ.
        let mut glue_dg_s = HashMap::new();
        glue_dg_s.insert(RefOrPair::Ref("p".into()), GsOrSeq::GS((-3.0, 0.0)));
        glue_dg_s.insert(RefOrPair::Ref("q".into()), GsOrSeq::GS((-3.0, 0.0)));
        let params = SDC2DParams {
            strands: vec![
                SDC2DStrand {
                    name: Some("P".into()),
                    color: None,
                    concentration: 1e-6,
                    west_glue: None,
                    north_glue: None,
                    east_glue: None,
                    south_glue: None,
                    bottom_glue: Some("p".into()),
                },
                SDC2DStrand {
                    name: Some("Q".into()),
                    color: None,
                    concentration: 1e-6,
                    west_glue: None,
                    north_glue: None,
                    east_glue: None,
                    south_glue: None,
                    bottom_glue: Some("q".into()),
                },
            ],
            scaffold: vec![vec![Some("p*".into()), Some("q*".into())]],
            scaffold_concentration: 1e-9,
            glue_dg_s,
            k_f: 1e6,
            temperature: 37.0,
            seed: vec![],
        };
        let sys = SDC2D::from_params(params);
        let p_id = sys.strand_names.iter().position(|n| n == "P").unwrap() as Tile;
        let q_id = sys.strand_names.iter().position(|n| n == "Q").unwrap() as Tile;
        assert_eq!(sys.friends_btm[(0, 0)], vec![p_id]);
        assert_eq!(sys.friends_btm[(0, 1)], vec![q_id]);
    }

    #[test]
    fn test_seed_recorded() {
        let mut params = minimal_2x2_params();
        params.seed = vec![(0, 0, "A".into())];
        let sys = SDC2D::from_params(params);
        assert!(sys.is_seed(&PointSafe2((0, 0))));
        assert!(!sys.is_seed(&PointSafe2((0, 1))));
    }

    use crate::canvas::{Canvas, CanvasSquare};
    use crate::state::{NullStateTracker, QuadTreeState, StateWithCreate};

    type TState = QuadTreeState<CanvasSquare, NullStateTracker>;

    fn make_state(sys: &SDC2D, n: usize) -> TState {
        TState::empty_with_types((n, n), sys.n_strands()).unwrap()
    }

    /// Build an `n x n` canvas-sized SDC2D with one strand A bound by a single
    /// glue g everywhere on the interior. Border positions get null scaffold
    /// glue (binding to nothing). Interior is `[2..n-2, 2..n-2]`.
    fn padded_uniform_sys(n: usize, dg: f64, ds: f64) -> SDC2D {
        let mut scaffold = vec![vec![None::<String>; n]; n];
        for row in scaffold.iter_mut().take(n - 2).skip(2) {
            for cell in row.iter_mut().take(n - 2).skip(2) {
                *cell = Some("g*".into());
            }
        }
        let mut glue_dg_s = HashMap::new();
        glue_dg_s.insert(RefOrPair::Ref("g".into()), GsOrSeq::GS((dg, ds)));
        SDC2D::from_params(SDC2DParams {
            strands: vec![SDC2DStrand {
                name: Some("A".into()),
                color: None,
                concentration: 1e-6,
                west_glue: None,
                north_glue: None,
                east_glue: None,
                south_glue: None,
                bottom_glue: Some("g".into()),
            }],
            scaffold,
            scaffold_concentration: 1e-9,
            glue_dg_s,
            k_f: 1e6,
            temperature: 37.0,
            seed: vec![],
        })
    }

    #[test]
    fn test_seed_does_not_detach() {
        let mut sys = padded_uniform_sys(8, -10.0, 0.0);
        // Pin A at (3, 3).
        sys.seed.insert(PointSafe2((3, 3)), 1);
        let mut state: TState = make_state(&sys, 8);
        state.set_sa(&PointSafe2((3, 3)), &1u32);
        let rate = sys.monomer_detachment_rate_at_point(&state, PointSafe2((3, 3)));
        assert_eq!(rate, PerSecond::zero());
        // A neighboring (non-seed) interior point with no tile should also have
        // a non-zero attachment rate, confirming the rest of the engine works.
        let att = sys.total_monomer_attachment_rate_at_point(&state, PointSafe2((4, 4)));
        assert!(f64::from(att) > 0.0);
    }

    #[test]
    #[should_panic(expected = "not supported")]
    fn test_no_fission_panic() {
        let sys = padded_uniform_sys(8, -5.0, 0.0);
        let mut state: TState = make_state(&sys, 8);
        let evt = Event::PolymerDetachment(vec![PointSafe2((3, 3))]);
        sys.update_after_event(&mut state, &evt);
    }

    #[test]
    fn test_corner_vs_interior_binding() {
        // Strand A binds itself on every edge with the same g/g* pair, so each
        // bond is identical. With no neighbors, only the scaffold bond
        // contributes; surround with neighbors, the full 5-bond sum applies.
        let dg = -2.0;
        let mut glue_dg_s = HashMap::new();
        glue_dg_s.insert(RefOrPair::Ref("g".into()), GsOrSeq::GS((dg, 0.0)));
        let mut scaffold = vec![vec![None::<String>; 8]; 8];
        for row in scaffold.iter_mut().take(6).skip(2) {
            for cell in row.iter_mut().take(6).skip(2) {
                *cell = Some("g*".into());
            }
        }
        let sys = SDC2D::from_params(SDC2DParams {
            strands: vec![SDC2DStrand {
                name: Some("A".into()),
                color: None,
                concentration: 1e-6,
                west_glue: Some("g".into()),
                north_glue: Some("g".into()),
                east_glue: Some("g*".into()),
                south_glue: Some("g*".into()),
                bottom_glue: Some("g".into()),
            }],
            scaffold,
            scaffold_concentration: 1e-9,
            glue_dg_s,
            k_f: 1e6,
            temperature: 37.0,
            seed: vec![],
        });
        let mut state: TState = make_state(&sys, 8);

        // Place a strand at an isolated interior point.
        let solo = PointSafe2((3, 3));
        state.set_sa(&solo, &1u32);
        let e_solo = sys.bond_energy_of_strand(&state, solo, 1);

        // Surround a different strand on all four sides.
        let center = PointSafe2((4, 4));
        for &p in &[
            PointSafe2((3, 4)),
            PointSafe2((5, 4)),
            PointSafe2((4, 3)),
            PointSafe2((4, 5)),
            center,
        ] {
            state.set_sa(&p, &1u32);
        }
        let e_full = sys.bond_energy_of_strand(&state, center, 1);

        // Per-edge β·ΔG is the same for every bond. Solo = 1 bond, full = 5.
        let per_bond = sys.bond_with_scaffold(3, 3, 1);
        assert!((e_solo - per_bond).abs() < 1e-12);
        assert!((e_full - 5.0 * per_bond).abs() < 1e-12);
    }

    #[test]
    fn test_independence_we_ns() {
        // Two independent glue families: "h" on W/E edges, "v" on N/S edges.
        // Strand binds "h" west + "h*" east, and "v" north + "v*" south.
        // Verify bond_energy_of_strand sums them with no cross-talk.
        let mut glue_dg_s = HashMap::new();
        glue_dg_s.insert(RefOrPair::Ref("h".into()), GsOrSeq::GS((-3.0, 0.0)));
        glue_dg_s.insert(RefOrPair::Ref("v".into()), GsOrSeq::GS((-7.0, 0.0)));
        glue_dg_s.insert(RefOrPair::Ref("g".into()), GsOrSeq::GS((-1.0, 0.0)));
        let mut scaffold = vec![vec![None::<String>; 8]; 8];
        for row in scaffold.iter_mut().take(6).skip(2) {
            for cell in row.iter_mut().take(6).skip(2) {
                *cell = Some("g*".into());
            }
        }
        let sys = SDC2D::from_params(SDC2DParams {
            strands: vec![SDC2DStrand {
                name: Some("A".into()),
                color: None,
                concentration: 1e-6,
                west_glue: Some("h".into()),
                north_glue: Some("v".into()),
                east_glue: Some("h*".into()),
                south_glue: Some("v*".into()),
                bottom_glue: Some("g".into()),
            }],
            scaffold,
            scaffold_concentration: 1e-9,
            glue_dg_s,
            k_f: 1e6,
            temperature: 37.0,
            seed: vec![],
        });
        let mut state: TState = make_state(&sys, 8);
        // Place a row of three: west, center, east. Center has W and E neighbors but no N/S.
        for &p in &[PointSafe2((4, 3)), PointSafe2((4, 4)), PointSafe2((4, 5))] {
            state.set_sa(&p, &1u32);
        }
        let center = PointSafe2((4, 4));
        let e_we_only = sys.bond_energy_of_strand(&state, center, 1);
        let scaffold_part = sys.bond_with_scaffold(4, 4, 1);
        let we_part = sys.bond_we(1, 1) * 2.0; // both W and E neighbor present
        assert!((e_we_only - (scaffold_part + we_part)).abs() < 1e-12);

        // Now also add N and S neighbors.
        for &p in &[PointSafe2((3, 4)), PointSafe2((5, 4))] {
            state.set_sa(&p, &1u32);
        }
        let e_full = sys.bond_energy_of_strand(&state, center, 1);
        let ns_part = sys.bond_ns(1, 1) * 2.0;
        assert!((e_full - (scaffold_part + we_part + ns_part)).abs() < 1e-12);

        // Sanity: W/E and N/S use different glue energies, so they can't be
        // accidentally collapsed.
        assert!((sys.bond_we(1, 1) - sys.bond_ns(1, 1)).abs() > 1e-6);
    }

    #[test]
    fn test_run_minimal_evolve() {
        // A small scaffold, briefly evolved, must accumulate at least one tile.
        let sys = padded_uniform_sys(8, -8.0, 0.0);
        let mut state: TState = make_state(&sys, 8);
        sys.update_state(&mut state, &NeededUpdate::All);
        let bounds = crate::system::EvolveBounds {
            for_events: Some(50),
            ..Default::default()
        };
        let _ = sys.evolve(&mut state, bounds).unwrap();
        assert!(state.calc_n_tiles() > 0);
    }
}
