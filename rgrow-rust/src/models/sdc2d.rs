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

use crate::base::{Glue, Tile};
use crate::canvas::PointSafe2;
use crate::colors::get_color_or_random;
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
}
