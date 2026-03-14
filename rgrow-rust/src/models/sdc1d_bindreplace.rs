/// The Bind-Replace model of SDC from the SI of https://www.biorxiv.org/content/10.1101/2025.07.16.664196v1
/// As this model is extremely simple, it also serves as a nice guide to implementing new models.
use core::panic;
use std::collections::HashMap;

use super::sdc1d::{gsorseq_to_gs, self_and_inverse, RefOrPair, SDCParams, SDCStrand};
#[allow(unused_imports)]
use crate::{
    canvas::{PointSafe2, PointSafeHere},
    colors::get_color_or_random,
    models::sdc1d::{get_or_generate, SingleOrMultiScaffold},
    state::State,
    system::{Event, System, TileBondInfo},
    units::{
        Celsius, Energy, KcalPerMol, KcalPerMolKelvin, Kelvin, Molar, PerMolarSecond, PerSecond,
        Temperature,
    },
};
#[allow(unused_imports)]
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use numpy::PyArrayMethods;
#[cfg(feature = "python")]
#[allow(unused_imports)]
use numpy::ToPyArray;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Glue(u64);

impl Glue {
    fn matches(&self, other: Glue) -> bool {
        self.0 != 0 && other.0 != 0 && (self.0 - 1) ^ (other.0 - 1) == 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
struct Tile(u32);

#[cfg_attr(feature = "python", pyclass(subclass, module = "rgrow.rgrow"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDC1DBindReplace {
    pub strand_names: Vec<String>,
    pub strand_colors: Vec<[u8; 4]>,
    pub glue_names: Vec<String>,
    pub scaffold: Vec<Glue>,
    pub strand_glues: Vec<(Glue, Glue, Glue)>,

    pub strand_concentrations: Array1<f64>,
    pub kf: PerMolarSecond,
    pub temperature: Celsius,
    pub account_for_energy: bool,
    pub allow_weak_replacement: bool,
    pub physical_attachment_rate: bool,
    /// When true (and `account_for_energy` is also true), the replacement rate
    /// at a filled site accounts for both the expected detachment time *and*
    /// the expected re-attachment time:
    ///   r = 1 / (1/r_detach + 1/r_attach)
    /// where r_detach = kf·exp(βΔG) and r_attach = Σ(kf·conc_i) over valid
    /// replacers.  When false, the replacement rate is just r_detach.
    pub bindunbind_replacement_rate: bool,
    /// When true, a tile can be "replaced" by the same tile type (effectively
    /// a detach-then-reattach cycle that wastes time without changing state).
    /// Combined with `bindunbind_replacement_rate`, this makes the bind-replace
    /// kinetics closely match the full SDC model.
    pub allow_same_replacement: bool,
    pub delta_g_matrix: Array2<KcalPerMol>,
    pub entropy_matrix: Array2<KcalPerMolKelvin>,

    #[serde(skip)]
    matching_tiles_at_site: Vec<Vec<Tile>>,
    #[serde(skip)]
    strand_energy_bonds: Array2<f64>,
    #[serde(skip)]
    scaffold_energy_bonds: Array2<f64>,
}

impl TileBondInfo for SDC1DBindReplace {
    fn tile_colors(&self) -> &Vec<[u8; 4]> {
        &self.strand_colors
    }

    fn tile_names(&self) -> &[String] {
        &self.strand_names
    }

    fn bond_names(&self) -> &[String] {
        &self.glue_names
    }
}

impl System for SDC1DBindReplace {
    fn system_info(&self) -> String {
        format!(
            "SDC1DBindReplace with {} strands, {} glues, and length {} scaffold",
            self.strand_names.len() - 1,
            self.glue_names.len() - 1,
            self.scaffold.len()
        )
    }

    fn update_after_event<St: State>(&self, state: &mut St, event: &Event) {
        match event {
            Event::None => todo!(),
            Event::MonomerAttachment(scaffold_point, _)
            | Event::MonomerDetachment(scaffold_point)
            | Event::MonomerChange(scaffold_point, _) => {
                self.update_monomer_point(state, scaffold_point)
            }
            _ => panic!("This event is not supported in SDC"),
        }
    }

    fn event_rate_at_point<St: crate::state::State>(
        &self,
        state: &St,
        p: crate::canvas::PointSafeHere,
    ) -> PerSecond {
        if !state.inbounds(p.0) {
            return PerSecond::new(0.0);
        };

        let coord = PointSafe2(p.0);
        match state.tile_at_point(coord) {
            0 => {
                let candidates = &self.matching_tiles_at_site[coord.0 .1];
                if candidates.is_empty() {
                    PerSecond::from(0.0)
                } else if self.physical_attachment_rate {
                    let total: f64 = candidates
                        .iter()
                        .map(|t| f64::from(self.kf) * self.strand_concentrations[t.0 as usize])
                        .sum();
                    PerSecond::from(total)
                } else {
                    PerSecond::from(1.0)
                }
            }
            s => {
                let (glue_w, _, glue_e) = self.strand_glues[s as usize];
                let (glue_to_e, _, _) = self.strand_glues[state.tile_to_e(coord) as usize];
                let (_, _, glue_to_w) = self.strand_glues[state.tile_to_w(coord) as usize];
                let ng = glue_w.matches(glue_to_w) as u8 + glue_e.matches(glue_to_e) as u8;

                let has_replacer =
                    self.matching_tiles_at_site[coord.0 .1]
                        .iter()
                        .any(|&possible_replace| {
                            if s == possible_replace.0 && !self.allow_same_replacement {
                                return false;
                            }
                            if self.allow_weak_replacement {
                                return true;
                            }
                            let (glue_w, _, glue_e) =
                                self.strand_glues[possible_replace.0 as usize];
                            let ng_replace =
                                glue_w.matches(glue_to_w) as u8 + glue_e.matches(glue_to_e) as u8;
                            ng_replace >= ng
                        });

                if !has_replacer {
                    PerSecond::from(0.0)
                } else if self.account_for_energy {
                    let bond_energy = self.bond_energy_of_strand(state, coord, s);
                    let r_detach = f64::from(self.kf) * bond_energy.exp();
                    if self.bindunbind_replacement_rate {
                        let r_attach: f64 = self.matching_tiles_at_site[coord.0 .1]
                            .iter()
                            .filter(|&&t| {
                                if s == t.0 && !self.allow_same_replacement {
                                    return false;
                                }
                                if self.allow_weak_replacement {
                                    return true;
                                }
                                let (glue_w, _, glue_e) = self.strand_glues[t.0 as usize];
                                let ng_replace = glue_w.matches(glue_to_w) as u8
                                    + glue_e.matches(glue_to_e) as u8;
                                ng_replace >= ng
                            })
                            .map(|t| f64::from(self.kf) * self.strand_concentrations[t.0 as usize])
                            .sum();
                        // Combined rate: 1 / (1/r_detach + 1/r_attach)
                        PerSecond::from((r_detach * r_attach) / (r_detach + r_attach))
                    } else {
                        PerSecond::from(r_detach)
                    }
                } else {
                    PerSecond::from(1.0)
                }
            }
        }
    }

    fn choose_event_at_point<St: crate::state::State>(
        &self,
        state: &St,
        p: crate::canvas::PointSafe2,
        acc: PerSecond,
    ) -> (crate::system::Event, f64) {
        let t = Tile(state.tile_at_point(p));
        let mut mut_acc = acc;

        match t {
            Tile(0) => {
                let candidates = &self.matching_tiles_at_site[p.0 .1];
                if self.physical_attachment_rate {
                    for t in candidates {
                        let rate = f64::from(self.kf) * self.strand_concentrations[t.0 as usize];
                        mut_acc -= PerSecond::from(rate);
                        if mut_acc.0 <= 0.0 {
                            return (Event::MonomerAttachment(p, t.0), rate);
                        }
                    }
                } else {
                    let total_conc: f64 = candidates
                        .iter()
                        .map(|t| self.strand_concentrations[t.0 as usize])
                        .sum();
                    for t in candidates {
                        let frac = self.strand_concentrations[t.0 as usize] / total_conc;
                        mut_acc -= PerSecond::from(frac);
                        if mut_acc.0 <= 0.0 {
                            return (Event::MonomerAttachment(p, t.0), frac);
                        }
                    }
                }
                panic!(
                    "Should have found an event to choose at point {:?} with acc {}",
                    p, acc
                );
            }
            t => {
                let (glue_w, _, glue_e) = self.strand_glues[t.0 as usize];
                let (glue_to_e, _, _) = self.strand_glues[state.tile_to_e(p) as usize];
                let (_, _, glue_to_w) = self.strand_glues[state.tile_to_w(p) as usize];
                let ng = glue_w.matches(glue_to_w) as u8 + glue_e.matches(glue_to_e) as u8;

                // Collect valid replacers
                let valid_replacers: Vec<Tile> = self.matching_tiles_at_site[p.0 .1]
                    .iter()
                    .copied()
                    .filter(|&possible_replace| {
                        if t.0 == possible_replace.0 && !self.allow_same_replacement {
                            return false;
                        }
                        if self.allow_weak_replacement {
                            return true;
                        }
                        let (glue_w, _, glue_e) = self.strand_glues[possible_replace.0 as usize];
                        let ng_replace =
                            glue_w.matches(glue_to_w) as u8 + glue_e.matches(glue_to_e) as u8;
                        ng_replace >= ng
                    })
                    .collect();

                let total_rate = if self.account_for_energy {
                    let r_detach = {
                        let bond_energy = self.bond_energy_of_strand(state, p, t.0);
                        f64::from(self.kf) * bond_energy.exp()
                    };
                    if self.bindunbind_replacement_rate {
                        let r_attach: f64 = valid_replacers
                            .iter()
                            .map(|r| f64::from(self.kf) * self.strand_concentrations[r.0 as usize])
                            .sum();
                        (r_detach * r_attach) / (r_detach + r_attach)
                    } else {
                        r_detach
                    }
                } else {
                    1.0
                };

                let total_conc: f64 = valid_replacers
                    .iter()
                    .map(|r| self.strand_concentrations[r.0 as usize])
                    .sum();

                for &replacer in &valid_replacers {
                    let frac =
                        total_rate * self.strand_concentrations[replacer.0 as usize] / total_conc;
                    mut_acc -= PerSecond::from(frac);
                    if mut_acc.0 <= 0.0 {
                        return (Event::MonomerChange(p, replacer.0), frac);
                    }
                }
                panic!(
                    "Should have found an event to choose at point {:?} with acc {}",
                    p, acc
                );
            }
        }
    }

    fn seed_locs(&self) -> Vec<(crate::canvas::PointSafe2, crate::base::Tile)> {
        Vec::default()
    }

    fn calc_mismatch_locations<St: crate::state::State>(
        &self,
        state: &St,
    ) -> ndarray::Array2<usize> {
        let mut mismatch_locations =
            ndarray::Array2::<usize>::zeros((state.nrows(), state.ncols()));

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

                let (_, _, glue_e) = self.strand_glues[t as usize];
                let (glue_w, _, _) = self.strand_glues[t as usize];

                let mm_e = ((te != 0) && {
                    let (glue_w_te, _, _) = self.strand_glues[te as usize];
                    !glue_e.matches(glue_w_te)
                }) as usize;
                let mm_w = ((tw != 0) && {
                    let (_, _, glue_e_tw) = self.strand_glues[tw as usize];
                    !glue_e_tw.matches(glue_w)
                }) as usize;

                mismatch_locations[(i, j)] = 4 * mm_e + mm_w;
            }
        }

        mismatch_locations
    }

    fn set_param(
        &mut self,
        name: &str,
        value: Box<dyn std::any::Any>,
    ) -> Result<crate::system::NeededUpdate, crate::base::GrowError> {
        use crate::base::GrowError;
        use crate::system::NeededUpdate;
        match name {
            "kf" => {
                let kf = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.kf = PerMolarSecond::from(*kf);
                self.update();
                Ok(NeededUpdate::NonZero)
            }
            "temperature" => {
                let temperature = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.temperature = Celsius(*temperature);
                self.update();
                Ok(NeededUpdate::NonZero)
            }
            _ => Err(GrowError::NoParameter(name.to_string())),
        }
    }

    fn get_param(&self, name: &str) -> Result<Box<dyn std::any::Any>, crate::base::GrowError> {
        match name {
            "kf" => Ok(Box::new(f64::from(self.kf))),
            "temperature" => Ok(Box::new(self.temperature.0)),
            _ => Err(crate::base::GrowError::NoParameter(name.to_string())),
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
                current_value: self.temperature.0,
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
}

impl SDC1DBindReplace {
    fn update_monomer_point<S: State>(&self, state: &mut S, scaffold_point: &PointSafe2) {
        let mut points = Vec::with_capacity(3);

        let pw = state.move_sa_w(*scaffold_point);
        if state.inbounds(pw.0) {
            points.push((pw, self.event_rate_at_point(state, pw)));
        }
        let pe = state.move_sa_e(*scaffold_point);
        if state.inbounds(pe.0) {
            points.push((pe, self.event_rate_at_point(state, pe)));
        }
        let ph = PointSafeHere(scaffold_point.0);
        points.push((ph, self.event_rate_at_point(state, ph)));
        state.update_multiple(&points);
    }

    /// Compute β·ΔG for a glue pair (energy in units of kT).
    fn glue_energy(&self, g1: Glue, g2: Glue) -> f64 {
        let dg = self.delta_g_matrix[(g1.0 as usize, g2.0 as usize)];
        let ds = self.entropy_matrix[(g1.0 as usize, g2.0 as usize)];
        // ΔG(T) = ΔG(37°C) - (T - 37°C) × ΔS, then multiply by β = 1/(RT)
        let temp_kelvin: Kelvin = self.temperature.into();
        let glue_value = dg - (temp_kelvin - Celsius(37.0)) * ds;
        glue_value.times_beta(self.temperature)
    }

    /// Precompute strand-strand and scaffold-strand energy bond arrays.
    fn fill_energy_arrays(&mut self) {
        let strand_count = self.strand_names.len();
        let scaffold_len = self.scaffold.len();

        self.strand_energy_bonds = Array2::zeros((strand_count, strand_count));
        self.scaffold_energy_bonds = Array2::zeros((scaffold_len, strand_count));

        // strand-strand: east glue of x bonds with west glue of y
        for x in 0..strand_count {
            let (_, _, glue_e_x) = self.strand_glues[x];
            for y in 0..strand_count {
                let (glue_w_y, _, _) = self.strand_glues[y];
                self.strand_energy_bonds[(x, y)] = self.glue_energy(glue_e_x, glue_w_y);
            }
        }

        // scaffold-strand: scaffold glue bonds with bottom glue of strand
        for (pos, &scaffold_glue) in self.scaffold.iter().enumerate() {
            for s in 0..strand_count {
                let (_, glue_b, _) = self.strand_glues[s];
                self.scaffold_energy_bonds[(pos, s)] = self.glue_energy(scaffold_glue, glue_b);
            }
        }
    }

    /// Sum of β·ΔG for all bonds of a strand at a given position.
    fn bond_energy_of_strand<S: State>(&self, state: &S, point: PointSafe2, strand: u32) -> f64 {
        let w = state.tile_to_w(point) as usize;
        let e = state.tile_to_e(point) as usize;
        let s = strand as usize;
        self.scaffold_energy_bonds[(point.0 .1, s)]
            + self.strand_energy_bonds[(w, s)]
            + self.strand_energy_bonds[(s, e)]
    }

    pub fn from_params(params: SDCParams) -> Self {
        let mut glue_name_map: HashMap<String, usize> = HashMap::new();
        let strand_count = params.strands.len() + 1; // to account for empty

        // Extract energy params before consuming strands
        let kf = PerMolarSecond::new(params.k_f);
        let temperature = Celsius(params.temperature);
        let glue_dg_s = params.glue_dg_s;
        let junction_penalty_dg = params.junction_penalty_dg;
        let junction_penalty_ds = params.junction_penalty_ds;

        let mut strand_names = Vec::with_capacity(strand_count);
        let mut strand_colors = Vec::with_capacity(strand_count);
        let mut strand_concs = Vec::with_capacity(strand_count);
        let mut glues = Vec::with_capacity(strand_count);
        let mut gluenum = 1;

        strand_names.push("empty".to_string());
        strand_colors.push([0, 0, 0, 0]);
        strand_concs.push(0.0);
        glues.push((Glue(0), Glue(0), Glue(0)));

        for (
            id,
            SDCStrand {
                name,
                color,
                concentration,
                btm_glue,
                left_glue,
                right_glue,
            },
        ) in params.strands.into_iter().enumerate()
        {
            strand_names.push(name.unwrap_or(format!("{}", id)));
            strand_colors.push(get_color_or_random(color.as_deref()).unwrap());
            strand_concs.push(concentration);

            let s_glues = (
                Glue(get_or_generate(&mut glue_name_map, &mut gluenum, left_glue) as u64),
                Glue(get_or_generate(&mut glue_name_map, &mut gluenum, btm_glue) as u64),
                Glue(get_or_generate(&mut glue_name_map, &mut gluenum, right_glue) as u64),
            );
            glues.push(s_glues);
        }

        let scaffold = match params.scaffold {
            SingleOrMultiScaffold::Single(s) => {
                let mut scaffold = Vec::<Glue>::with_capacity(s.len());
                for g in s.into_iter() {
                    scaffold.push(Glue(
                        get_or_generate(&mut glue_name_map, &mut gluenum, g) as u64
                    ));
                }
                scaffold
            }
            SingleOrMultiScaffold::Multi(_m) => todo!(),
        };

        // Build energy matrices from glue_dg_s using glue_name_map (before consuming it)
        let mut delta_g_matrix = Array2::<KcalPerMol>::zeros((gluenum, gluenum));
        let mut entropy_matrix = Array2::<KcalPerMolKelvin>::zeros((gluenum, gluenum));

        for (k, gs_or_dna_sequence) in glue_dg_s.iter() {
            let gs = gsorseq_to_gs(gs_or_dna_sequence);

            let (i, j) = match k {
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

            let (i, j) = match (glue_name_map.get(&i), glue_name_map.get(&j)) {
                (Some(&x), Some(&y)) => (x, y),
                _ => continue,
            };

            delta_g_matrix[[i, j]] = gs.0 + junction_penalty_dg.unwrap_or(KcalPerMol(0.0));
            delta_g_matrix[[j, i]] = gs.0 + junction_penalty_dg.unwrap_or(KcalPerMol(0.0));
            entropy_matrix[[i, j]] = gs.1 + junction_penalty_ds.unwrap_or(KcalPerMolKelvin(0.0));
            entropy_matrix[[j, i]] = gs.1 + junction_penalty_ds.unwrap_or(KcalPerMolKelvin(0.0));
        }

        // Now consume glue_name_map to build glue_names
        let mut glue_names = vec![String::default(); gluenum];
        for (sx, i) in glue_name_map.into_iter() {
            glue_names[i] = sx;
        }

        let mut sys = SDC1DBindReplace {
            strand_names,
            strand_colors,
            glue_names,
            scaffold: scaffold.to_vec(),
            strand_glues: glues,
            strand_concentrations: Array1::from_vec(strand_concs),
            kf,
            temperature,
            account_for_energy: false,
            allow_weak_replacement: false,
            physical_attachment_rate: false,
            bindunbind_replacement_rate: false,
            allow_same_replacement: false,
            delta_g_matrix,
            entropy_matrix,
            matching_tiles_at_site: vec![vec![]; scaffold.len()],
            strand_energy_bonds: Array2::zeros((strand_count, strand_count)),
            scaffold_energy_bonds: Array2::zeros((scaffold.len(), strand_count)),
        };
        sys.update();
        sys
    }

    pub fn update(&mut self) {
        for (i, &scaffold_glue) in self.scaffold.iter().enumerate() {
            let mut matching_tiles = vec![];
            for (tile_num, &(_glue_w, glue_b, _glue_e)) in self.strand_glues.iter().enumerate() {
                if glue_b.matches(scaffold_glue) && self.strand_concentrations[tile_num] > 0.0 {
                    matching_tiles.push(Tile(tile_num as u32));
                }
            }
            self.matching_tiles_at_site[i] = matching_tiles;
        }

        if self.account_for_energy {
            self.fill_energy_arrays();
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl SDC1DBindReplace {
    #[new]
    fn py_new(params: SDCParams) -> Self {
        Self::from_params(params)
    }

    #[getter]
    fn get_account_for_energy(&self) -> bool {
        self.account_for_energy
    }

    #[setter]
    fn set_account_for_energy(&mut self, value: bool) {
        self.account_for_energy = value;
        self.update();
    }

    #[getter]
    fn get_allow_weak_replacement(&self) -> bool {
        self.allow_weak_replacement
    }

    #[setter]
    fn set_allow_weak_replacement(&mut self, value: bool) {
        self.allow_weak_replacement = value;
    }

    #[getter]
    fn get_physical_attachment_rate(&self) -> bool {
        self.physical_attachment_rate
    }

    #[setter]
    fn set_physical_attachment_rate(&mut self, value: bool) {
        self.physical_attachment_rate = value;
    }

    #[getter]
    fn get_bindunbind_replacement_rate(&self) -> bool {
        self.bindunbind_replacement_rate
    }

    #[setter]
    fn set_bindunbind_replacement_rate(&mut self, value: bool) {
        self.bindunbind_replacement_rate = value;
    }

    #[getter]
    fn get_allow_same_replacement(&self) -> bool {
        self.allow_same_replacement
    }

    #[setter]
    fn set_allow_same_replacement(&mut self, value: bool) {
        self.allow_same_replacement = value;
    }
}
