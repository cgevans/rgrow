macro_rules! type_alias {
    ($($t:ty => $($i:ident),*);* $(;)?) => {
        $($(type $i = $t;)*)*
    };
}

/*
* Important Notes
*
* Given some PointSafe2, in this model, it will represnt two things
* 1. Which of the scaffolds has an event happening
* 2. In which position of the scaffold said event will take place
*
* TODO:
* - There are quite a few expects that need to be handled better
* */

use core::f64;
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::OnceLock,
};

use astro_float::{BigFloat, RoundingMode, Sign};
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    base::{Glue, GrowError, Rate, Tile},
    canvas::{PointSafe2, PointSafeHere},
    colors::get_color_or_random,
    state::{State, StateEnum},
    system::{Event, EvolveBounds, NeededUpdate, System, TileBondInfo},
};

use ndarray::prelude::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use numpy::ToPyArray;
#[cfg(feature = "python")]
use pyo3::prelude::*;

type_alias!( f64 => Strength, RatePerConc, Conc );

const WEST_GLUE_INDEX: usize = 0;
const BOTTOM_GLUE_INDEX: usize = 1;
const EAST_GLUE_INDEX: usize = 2;
const R: f64 = 1.98720425864083 / 1000.0; // in kcal/mol/K
const U0: f64 = 1.0;

fn bigfloat_to_f64(big_float: &BigFloat, rounding_mode: RoundingMode) -> f64 {
    let mut big_float = big_float.clone();
    big_float.set_precision(64, rounding_mode).unwrap();
    let sign = big_float.sign().unwrap();
    let exponent = big_float.exponent().unwrap();
    let mantissa = big_float.mantissa_digits().unwrap()[0];
    if mantissa == 0 {
        return 0.0;
    }
    let mut exponent: isize = exponent as isize + 0b1111111111;
    let mut ret = 0;
    if exponent >= 0b11111111111 {
        match sign {
            Sign::Pos => f64::INFINITY,
            Sign::Neg => f64::NEG_INFINITY,
        }
    } else if exponent <= 0 {
        let shift = -exponent;
        if shift < 52 {
            ret |= mantissa >> (shift + 12);
            if sign == Sign::Neg {
                ret |= 0x8000000000000000u64;
            }
            f64::from_bits(ret)
        } else {
            0.0
        }
    } else {
        let mantissa = mantissa << 1;
        exponent -= 1;
        if sign == Sign::Neg {
            ret |= 1;
        }
        ret <<= 11;
        ret |= exponent as u64;
        ret <<= 52;
        ret |= mantissa >> 12;
        f64::from_bits(ret)
    }
}

#[cfg_attr(feature = "python", pyclass(subclass))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDC {
    /// The anchor tiles for each of the scaffolds
    ///
    /// To get the anchor tile of the nth scaffold, anchor_tiles.get(n)
    pub anchor_tiles: Vec<(PointSafe2, Tile)>,
    pub strand_names: Vec<String>,
    pub glue_names: Vec<String>,
    /// Colors of the scaffolds, strands can only stick if the
    /// colors are a perfect match
    ///
    /// Note that this system will accept many scaffolds, thus this is a  2d array and not a 1d
    /// array
    pub scaffold: Array2<Glue>,
    /// All strands in the system, they are represented by tiles
    /// with only glue on the south, west, and east (nothing can stuck to the top of a strand)
    // pub strands: Array1<Tile>,
    pub strand_concentration: Array1<Conc>,
    /// The concentration of the scaffold
    pub scaffold_concentration: Conc,
    /// Glues of a given strand by id
    ///
    /// Note that the glues will be sorted in the following manner:
    /// [
    ///     (0) -- [left glue, bottom glue, right glue]
    ///     ...
    ///     (n) -- [left glue, bottom glue, right glue]
    /// ]
    pub glues: Array2<Glue>,
    /// Each strand will be given a color so that it can be easily identified
    /// when illustrated
    pub colors: Vec<[u8; 4]>,
    /// The (de)attachment rates will depend on this constant(for the system) value
    pub kf: RatePerConc,
    /// FIXME: Change this to a vector to avoid hashing time
    ///
    /// Set of tiles that can stick to scaffold gap with a given glue
    pub friends_btm: HashMap<Glue, HashSet<Tile>>,
    /// Delta G at 37 degrees C in the formula to genereate the glue strengths
    pub delta_g_matrix: Array2<f64>,
    /// S in the formula to geenrate the glue strengths
    pub entropy_matrix: Array2<f64>,
    /// Temperature of the system
    ///
    /// Not pub so that it cant accidentally be changed other than with the setter function
    /// that will also recalculate energy arrays
    temperature: f64,
    /// The energy with which two strands will bond
    ///
    /// This array is indexed as follows. Given strands x and y, where x is to the west of y
    /// (meaning that the east of x forms a bond with the west of y), the energy of said bond
    /// is given by energy_bonds[(x, y)]
    #[serde(skip)]
    strand_energy_bonds: Array2<OnceLock<f64>>,
    /// The energy with which a strand attached to scaffold
    #[serde(skip)]
    scaffold_energy_bonds: Array1<OnceLock<f64>>,
}

impl SDC {
    fn bond_between_strands(&self, x: Tile, y: Tile) -> f64 {
        *self.strand_energy_bonds[(x as usize, y as usize)].get_or_init(|| {
            let x_east_glue = self.glues[(x as usize, EAST_GLUE_INDEX)];
            let y_west_glue = self.glues[(y as usize, WEST_GLUE_INDEX)];
            let glue_value = &self.delta_g_matrix[(x_east_glue, y_west_glue)]
                - (self.temperature - 37.0) * &self.entropy_matrix[(x_east_glue, y_west_glue)];
            glue_value / self.rtval()
        })
    }

    fn bond_with_scaffold(&self, x: Tile) -> f64 {
        *self.scaffold_energy_bonds[x as usize].get_or_init(|| {
            let x_bmt = self.glues[(x as usize, BOTTOM_GLUE_INDEX)];
            if x_bmt == 0 {
                return 0.0;
            }

            let x_inv = if x_bmt % 2 == 1 { x_bmt + 1 } else { x_bmt - 1 };
            let glue_value = &self.delta_g_matrix[(x_bmt, x_inv)]
                - (self.temperature - 37.0) * &self.entropy_matrix[(x_bmt, x_inv)];
            glue_value / self.rtval()
        })
    }

    fn new(
        anchor_tiles: Vec<(PointSafe2, Tile)>,
        strand_names: Vec<String>,
        glue_names: Vec<String>,
        scaffold: Array2<Glue>,
        strand_concentration: Array1<Conc>,
        scaffold_concentration: Conc,
        glues: Array2<Glue>,
        colors: Vec<[u8; 4]>,
        kf: RatePerConc,
        delta_g_matrix: Array2<f64>,
        entropy_matrix: Array2<f64>,
        temperature: f64,
    ) -> SDC {
        let strand_count = strand_names.len();
        let mut s = SDC {
            anchor_tiles,
            strand_concentration,
            strand_names,
            colors,
            glues,
            scaffold,
            glue_names,
            kf,
            delta_g_matrix,
            entropy_matrix,
            temperature,
            scaffold_concentration,
            // These will be generated by the update_system function next, so just leave them
            // empty for now
            friends_btm: HashMap::new(),
            strand_energy_bonds: Array2::default((strand_count, strand_count)),
            scaffold_energy_bonds: Array1::default(strand_count),
        };
        s.update_system();
        s
    }

    fn update_system(&mut self) {
        self.empty_cache();
        self.generate_friends();
    }

    fn empty_cache(&mut self) {
        let strand_count = self.strand_names.len();
        self.strand_energy_bonds = Array2::default((strand_count, strand_count));
        self.scaffold_energy_bonds = Array1::default(strand_count);
    }

    fn generate_friends(&mut self) {
        let mut friends_btm = HashMap::new();
        for (t, &b) in self
            .glues
            .index_axis(ndarray::Axis(1), BOTTOM_GLUE_INDEX)
            .indexed_iter()
        {
            // 0 <-> Nothing
            // 1 <-> 2
            // 3 <-> 4
            // ...

            if b == 0 {
                continue;
            }

            let b_inverse = if b % 2 == 1 { b + 1 } else { b - 1 };
            friends_btm
                .entry(b_inverse)
                .or_insert(HashSet::new())
                .insert(t as u32);
        }
        self.friends_btm = friends_btm;
    }

    pub fn change_temperature_to(&mut self, celsius: f64) {
        self.temperature = celsius;
        self.update_system();
    }

    pub fn n_scaffolds<S: State>(&self, state: &S) -> usize {
        state.nrows_usable()
    }

    // FIXME:
    // MAKE SURE THAT THIS FUNCTION IS CORRECT
    //
    // It should count how many of a tile there is overall (attached or not)
    // ie monomer count
    //
    // count_monomer = (c_monomer / c_scaffold) * count_scaffold
    pub fn total_tile_count<S: State>(&self, state: &S, tile: Tile) -> usize {
        let per = self.strand_concentration[tile as usize] / self.scaffold_concentration;
        let net = per * self.n_scaffolds(state) as f64;
        net as usize
    }

    #[inline(always)]
    fn rtval(&self) -> f64 {
        R * (self.temperature + 273.15)
    }

    fn update_monomer_point<S: State>(&self, state: &mut S, scaffold_point: &PointSafe2) {
        let points = [
            state.move_sa_w(*scaffold_point),
            state.move_sa_e(*scaffold_point),
            PointSafeHere(scaffold_point.0),
        ]
        .map(|point| (point, self.event_rate_at_point(state, point)));

        state.update_multiple(&points);
    }

    /// Fill the energy_bonds array
    fn fill_energy_array(&mut self) {
        let num_of_strands = self.strand_names.len();
        let glue_links = &self.delta_g_matrix - (self.temperature - 37.0) * &self.entropy_matrix;
        // For each *possible* pair of strands, calculate the energy bond
        for strand_f in 1..(num_of_strands as usize) {
            // 1: no point in calculating for 0
            let (f_west_glue, f_btm_glue, f_east_glue) = {
                let glues = self.glues.row(strand_f);
                (
                    glues[WEST_GLUE_INDEX],
                    glues[BOTTOM_GLUE_INDEX],
                    glues[EAST_GLUE_INDEX],
                )
            };

            for strand_s in 0..(num_of_strands as usize) {
                let (s_west_glue, s_east_glue) = {
                    let glues = self.glues.row(strand_s);
                    (glues[WEST_GLUE_INDEX], glues[EAST_GLUE_INDEX])
                };

                // Calculate the energy between the two strands

                // Case 1: First strands is to the west of second
                // strand_f    strand_s
                let _ = self.strand_energy_bonds[(strand_f, strand_s)]
                    .set(glue_links[(f_east_glue, s_west_glue)] / self.rtval());

                // Case 2: First strands is to the east of second
                // strand_s    strand_f
                let _ = self.strand_energy_bonds[(strand_s, strand_f)]
                    .set(glue_links[(f_west_glue, s_east_glue)] / self.rtval());
            }

            // I suppose maybe we'd have weird strands with no position domain?
            if f_btm_glue == 0 {
                continue;
            }

            let b_inverse = if f_btm_glue % 2 == 1 {
                f_btm_glue + 1
            } else {
                f_btm_glue - 1
            };

            // Calculate the binding strength of the strand with the scaffold
            let _ = self.scaffold_energy_bonds[strand_f]
                .set(glue_links[(f_btm_glue, b_inverse)] / self.rtval());
        }
    }

    pub fn monomer_detachment_rate_at_point<S: State>(
        &self,
        state: &S,
        scaffold_point: PointSafe2,
    ) -> Rate {
        let strand = state.tile_at_point(scaffold_point);

        // let anchor_tile = self.anchor_tiles[(scaffold_point.0).0]; // FIXME: disabled anchor tiles for now

        // If we are trying to detach the anchor tile
        // There is no strand, thus nothing to be detached
        if strand == 0
        /*|| anchor_tile.0 == scaffold_point */
        {
            // FIXME: disabled anchor tiles for now
            return 0.0;
        }

        let bond_energy = self.bond_energy_of_strand(state, scaffold_point, strand);
        self.kf * bond_energy.exp()
    }

    pub fn choose_monomer_attachment_at_point<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: Rate,
    ) -> (bool, Rate, Event) {
        self.find_monomer_attachment_possibilities_at_point(state, acc, point, false)
    }

    pub fn choose_monomer_detachment_at_point<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        mut acc: Rate,
    ) -> (bool, Rate, Event) {
        acc -= self.monomer_detachment_rate_at_point(state, point);

        if acc > 0.0 {
            return (false, acc, Event::None);
        }

        (true, acc, Event::MonomerDetachment(point))
    }

    /// |      x y z <- attached strands (potentially empty)
    /// |_ _ _ _ _ _ _ _ _ _  <- Scaffold
    /// |        ^ point
    ///
    fn find_monomer_attachment_possibilities_at_point<S: State>(
        &self,
        state: &S,
        mut acc: Rate,
        scaffold_coord: PointSafe2,
        just_calc: bool,
    ) -> (bool, Rate, Event) {
        let point = scaffold_coord.into();
        let tile = state.tile_at_point(point);

        // If the scaffold already has a strand binded, then nothing can attach to it
        if tile != 0 {
            return (false, acc, Event::None);
        }

        let scaffold_glue = self
            .scaffold
            .get((point.0 .0.rem_euclid(self.scaffold.dim().0), point.0 .1))
            .expect("Invalid Index");

        let empty_map = HashSet::default();
        let friends = self.friends_btm.get(scaffold_glue).unwrap_or(&empty_map);
        let mut rand_thread = rand::thread_rng();

        for &strand in friends {
            acc -= self.kf * self.strand_concentration[strand as usize];
            if acc <= 0.0 && (!just_calc) {
                let rand: f64 = rand_thread.gen();
                let total = self.total_tile_count(state, strand) as f64;
                let attached = state.count_of_tile(strand) as f64;
                if rand <= attached / total {
                    return (true, acc, Event::None);
                }

                return (true, acc, Event::MonomerAttachment(point, strand));
            }
        }

        (false, acc, Event::None)
    }

    fn total_monomer_attachment_rate_at_poin<S: State>(
        &self,
        state: &S,
        scaffold_coord: PointSafe2,
    ) -> f64 {
        // If we set acc = 0, would it not be the case that we just attach to the first tile we can
        // ?
        match self.find_monomer_attachment_possibilities_at_point(state, 0.0, scaffold_coord, true)
        {
            (false, acc, _) => -acc,
            _ => panic!(),
        }
    }

    /// Get the sum of the energies of the bonded strands (if any)
    fn bond_energy_of_strand<S: State>(
        &self,
        state: &S,
        scaffold_point: PointSafe2,
        strand: u32,
    ) -> f64 {
        let (w, e) = (
            state.tile_to_w(scaffold_point) as usize,
            state.tile_to_e(scaffold_point) as usize,
        );

        self.bond_with_scaffold(strand)
            + self.bond_between_strands(strand, e as Tile)
            + self.bond_between_strands(w as Tile, strand)
    }

    fn scaffold(&self) -> Vec<usize> {
        self.scaffold.row(0).to_vec()
    }

    /// Given an SDC system, and some scaffold attachments
    ///
    /// 0 := nothing attached to the saccffold
    fn g_system(&self, attachments: &Vec<u32>) -> f64 {
        let mut sumg = 0.0;

        for (id, strand) in attachments.iter().enumerate() {
            if strand == &0 {
                continue;
            }

            // Add the energy of the strand and the scaffold
            sumg += self.bond_with_scaffold(*strand);
            if let Some(s) = attachments.get(id + 1) {
                // Also add the energy between the strand and the one to its right
                sumg += self.bond_between_strands(*strand, *s)
            };

            // Take into account the penalty
            let penalty = (self.strand_concentration[*strand as usize] / U0).ln();

            sumg -= penalty;
        }
        sumg * self.rtval()
    }

    // This is quite inefficient -- and clones a lot. If the scaffold were to be
    // longer than 10, this would not work
    pub fn system_states(&self) -> Vec<Vec<u32>> {
        let scaffold = self.scaffold();

        // Calculate the number of combinations ( this will i think make it a little more optimized
        // since we wont need realloc )
        let mut acc = 1;
        for b in &scaffold {
            if let Some(x) = self.friends_btm.get(&b) {
                // number of possible times + none
                acc *= x.len() + 1;
            }
        }

        let mut possible_scaffolds: Vec<Vec<u32>> = Vec::with_capacity(acc);
        possible_scaffolds.push(Vec::default());

        for b in &scaffold {
            let def = HashSet::default();
            let friends = self.friends_btm.get(b).unwrap_or(&def);

            possible_scaffolds = possible_scaffolds
                .iter()
                .flat_map(|scaffold_attachments| {
                    let mut new_combinations: Vec<Vec<u32>> = Vec::new();

                    // Each one of the friends will make one possible state
                    for f in friends {
                        let mut comb = scaffold_attachments.clone();
                        comb.push(*f);
                        new_combinations.push(comb);
                    }

                    // Also if nothing attached
                    let mut comb = scaffold_attachments.clone();
                    comb.push(0);
                    new_combinations.push(comb);
                    new_combinations
                })
                .collect();
        }

        possible_scaffolds
    }

    pub fn boltzman_function(&self, attachments: &Vec<u32>) -> f64 {
        let g_a = self.g_system(attachments);
        (-g_a / self.rtval()).exp()
    }

    pub fn partition_function_full(&self) -> f64 {
        self.system_states()
            .iter()
            .map(|attachments| self.boltzman_function(attachments))
            .sum()
    }

    ///
    /// Notes:
    /// - This only works for a single scaffold type.
    pub fn partition_function_fast(&self) -> f64 {
        let scaffold = self.scaffold();

        let max_competition = scaffold
            .iter()
            .map(|x| self.friends_btm.get(x).map(|y| y.len()).unwrap_or(0))
            .max()
            .unwrap();

        let mut z_curr = Array1::zeros(max_competition);
        let mut z_prev = Array1::zeros(max_competition);
        let mut z_sum = 1.0;
        let mut sum_a = 0.0;

        for (i, b) in scaffold.iter().enumerate() {
            // This is the partial partition function assuming that the previous site is empty:
            // it sums previous, previous partition functions (location i-2).
            sum_a += z_prev.sum();

            // We now move the previous (location i-1) location partial partition functions to the previous
            // array, and reset the current arry.
            z_prev.assign(&z_curr);
            z_curr.fill(0.);

            let friends = match self.friends_btm.get(b) {
                Some(f) => f,
                None => continue,
            };

            // Iterating through each possible attachment at the current location.
            for (j, &f) in friends.iter().enumerate() {
                let attachment_beta_dg =
                    self.bond_with_scaffold(f) - (self.strand_concentration[f as usize] / U0).ln();

                let t1 = (-attachment_beta_dg).exp();

                if i == 0 {
                    // First scaffold site.
                    // The partition function, given f attached at j, is all we need to calculate.
                    // z_sum has 1 in it right now, which covers the case where nothing is attached.
                    // sum_a has 0, because it is not being used yet.
                    z_curr[j] = t1;
                } else {
                    // Every other scaffold site
                    // t2 will hold the different cases where side i-1 has tile g in it.
                    let mut t2 = 0.;

                    match self.friends_btm.get(&scaffold[i - 1]) {
                        Some(ff) => {
                            for (k, &g) in ff.iter().enumerate() {
                                let left_beta_dg = self.bond_between_strands(g, f);
                                t2 += z_prev[k] * (-left_beta_dg).exp();
                            }
                        }
                        None => {}
                    }

                    // 1.0 -> *only* tile f is attached at position i.
                    // sum_a -> tile f is at position i, no tile is at position i-1.
                    // t2 -> tile f is at position i, another tile is at position i-1.
                    z_curr[j] = t1 * (1.0 + t2 + sum_a);
                }
                z_sum += z_curr[j];
            }
        }

        z_sum
    }

    ///
    /// Notes:
    /// - This only works for a single scaffold type.
    pub fn big_partition_function_fast(&self) -> BigFloat {
        let scaffold = self.scaffold();

        let prec = 64;
        let rm = astro_float::RoundingMode::None;
        let mut cc =
            astro_float::Consts::new().expect("An error occured when initializing constants");
        // let ctx = astro_float::ctx::Context::new(PREC, RM, cc, -100000, 100000);

        let max_competition = scaffold
            .iter()
            .map(|x| self.friends_btm.get(x).map(|y| y.len()).unwrap_or(0))
            .max()
            .unwrap();

        let mut z_curr = Array1::from_elem(max_competition, BigFloat::from_i32(0, prec));
        let mut z_prev = Array1::from_elem(max_competition, BigFloat::from_i32(0, prec));
        let mut z_sum = BigFloat::from_i64(1, prec);
        let mut sum_a = BigFloat::from_i64(0, prec);

        for (i, b) in scaffold.iter().enumerate() {
            // This is the partial partition function assuming that the previous site is empty:
            // it sums previous, previous partition functions (location i-2).
            for v in z_prev.iter() {
                sum_a = sum_a.add(v, prec, rm);
            }

            // We now move the previous (location i-1) location partial partition functions to the previous
            // array, and reset the current arry.
            z_prev.assign(&z_curr);
            z_curr.fill(BigFloat::from_i32(0, prec));

            let friends = match self.friends_btm.get(b) {
                Some(f) => f,
                None => continue,
            };

            // Iterating through each possible attachment at the current location.
            for (j, &f) in friends.iter().enumerate() {
                let attachment_beta_dg =
                    self.bond_with_scaffold(f) - (self.strand_concentration[f as usize] / U0).ln();

                let t1 = BigFloat::from_f64(-attachment_beta_dg, prec).exp(prec, rm, &mut cc);

                if i == 0 {
                    // First scaffold site.
                    // The partition function, given f attached at j, is all we need to calculate.
                    // z_sum has 1 in it right now, which covers the case where nothing is attached.
                    // sum_a has 0, because it is not being used yet.
                    z_curr[j] = t1;
                } else {
                    // Every other scaffold site
                    // t2 will hold the different cases where side i-1 has tile g in it.
                    let mut t2 = BigFloat::from_f64(0., prec);

                    match self.friends_btm.get(&scaffold[i - 1]) {
                        Some(ff) => {
                            for (k, &g) in ff.iter().enumerate() {
                                let left_beta_dg = self.bond_between_strands(g, f);
                                t2 = t2.add(
                                    &BigFloat::from_f64(-left_beta_dg, prec)
                                        .exp(prec, rm, &mut cc)
                                        .mul(&z_prev[k], prec, rm),
                                    prec,
                                    rm,
                                );
                            }
                        }
                        None => {}
                    }

                    // 1.0 -> *only* tile f is attached at position i.
                    // sum_a -> tile f is at position i, no tile is at position i-1.
                    // t2 -> tile f is at position i, another tile is at position i-1.
                    z_curr[j] = t1.mul(
                        &t2.add(&BigFloat::from_i64(1, prec), prec, rm)
                            .add(&sum_a, prec, rm),
                        prec,
                        rm,
                    );
                }
                z_sum = z_sum.add(&z_curr[j], prec, rm);
            }
        }
        z_sum
    }

    pub fn log_big_partition_function_fast(&self) -> f64 {
        let prec = 64;
        let rm = astro_float::RoundingMode::None;
        let mut cc =
            astro_float::Consts::new().expect("An error occured when initializing constants"); // FIXME: don't keep making this
        bigfloat_to_f64(
            &self.big_partition_function_fast().ln(prec, rm, &mut cc),
            rm,
        )
    }

    pub fn partition_function(&self) -> f64 {
        self.partition_function_fast()
    }

    pub fn probability_of_state_full(&self, system: &Vec<u32>) -> f64 {
        let sum_z = self.partition_function_fast();
        let this_system = self.boltzman_function(system);
        this_system / sum_z
    }

    pub fn probability_of_state(&self, system: &Vec<u32>) -> f64 {
        (-self.g_system(system) / self.rtval() - self.log_big_partition_function_fast()).exp()
    }
}

// MFE of system
// FIXME: Hashset needs some sort of ordering (by tile id? Will that be consistent between runs?)
impl SDC {
    /// Given some set of strands xi (see the graph below), and some tile for the
    /// y position, find the best match
    ///
    ///    x2
    ///    x1
    /// __ x0 y __
    ///
    /// Ideal bond = x1 y
    ///
    /// Return energy in the ideal case
    fn best_energy_for_right_strand(&self, left_possible: &Vec<(f64, Tile)>, right: &Tile) -> f64 {
        if left_possible.is_empty() {
            return self.bond_with_scaffold(*right);
        }

        left_possible
            .iter()
            .fold(f64::MAX, |acc, &(lenergy, left)| {
                let nenergy = lenergy + self.bond_between_strands(left, *right);
                acc.min(nenergy)
            })
            + self.bond_with_scaffold(*right)
    }

    /// This is for the standard case where the acc is not empty and the friends here hashset is
    /// not empty
    fn mfe_next_vector(
        &self,
        acc: Vec<(f64, Tile)>,
        friends_here: &HashSet<Tile>,
    ) -> Vec<(f64, Tile)> {
        friends_here
            .iter()
            .map(|tile| (self.best_energy_for_right_strand(&acc, tile), *tile))
            .collect()
    }

    /// Next vector in the case that the accumulator is empty (meaning this is the first set of
    /// strand in the system, in a system with strands everywhere, this will be the 3rd index)
    fn mfe_next_vector_empty_case(&self, friends_here: &HashSet<Tile>) -> Vec<(f64, Tile)> {
        friends_here
            .iter()
            .map(|&tile| (self.bond_with_scaffold(tile), tile))
            .collect()
    }

    fn mfe(&self) {
        self.scaffold()
            .iter()
            .fold(vec![], |acc, glue| match self.friends_btm.get(glue) {
                Some(friends_here) if !acc.is_empty() => self.mfe_next_vector(acc, friends_here),
                Some(friends_here) => self.mfe_next_vector_empty_case(friends_here),
                None => acc.into_iter().map(|(lenergy, _)| (lenergy, 0)).collect(),
            });
    }
}

impl System for SDC {
    fn update_after_event<St: State>(&self, state: &mut St, event: &Event) {
        match event {
            Event::None => todo!(),
            Event::MonomerAttachment(scaffold_point, _)
            | Event::MonomerDetachment(scaffold_point) => {
                self.update_monomer_point(state, scaffold_point)
            }
            _ => panic!("This event is not supported in SDC"),
        }
    }

    fn perform_event<St: State>(&self, state: &mut St, event: &Event) -> &Self {
        match event {
            Event::None => panic!("Being asked to perform null event."),
            Event::MonomerAttachment(point, strand) => {
                state.update_attachment(*strand);
                state.set_sa(point, strand);
            }
            Event::MonomerDetachment(point) => {
                let strand = state.tile_at_point(*point);
                state.update_detachment(strand);
                state.set_sa(point, &0);
            }
            _ => panic!("This event is not supported in SDC"),
        };
        self
    }

    fn event_rate_at_point<St: State>(
        &self,
        state: &St,
        p: crate::canvas::PointSafeHere,
    ) -> crate::base::Rate {
        if !state.inbounds(p.0) {
            return 0.0;
        }

        let scaffold_coord = PointSafe2(p.0);
        match state.tile_at_point(scaffold_coord) as u32 {
            // If the tile is empty, we will return the rate at which attachment can occur
            0 => self.total_monomer_attachment_rate_at_poin(state, scaffold_coord),
            // If the tile is full, we will return the rate at which detachment can occur
            _ => self.monomer_detachment_rate_at_point(state, scaffold_coord),
        }
    }

    fn choose_event_at_point<St: State>(
        &self,
        state: &St,
        point: crate::canvas::PointSafe2,
        acc: crate::base::Rate,
    ) -> crate::system::Event {
        match self.choose_monomer_detachment_at_point(state, point, acc) {
            (true, _, event) => event,
            (false, acc, _) => match self.choose_monomer_attachment_at_point(state, point, acc) {
                (true, _, event) => event,
                (false, acc, _) => panic!(
                    "Rate: {:?}, {:?}, {:?}, {:?}",
                    acc,
                    point,
                    state,
                    state.raw_array()
                ),
            },
        }
    }

    fn seed_locs(&self) -> Vec<(crate::canvas::PointSafe2, Tile)> {
        self.anchor_tiles.clone()
    }

    fn calc_mismatch_locations<St: State>(&self, state: &St) -> Array2<usize> {
        let threshold = -0.1; // Todo: fix this
        let mut mismatch_locations = Array2::<usize>::zeros((state.nrows(), state.ncols()));

        // TODO: this should use an iterator from the canvas, which we should implement.
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

                let mm_e = ((te != 0) & (self.bond_between_strands(t, te) > threshold)) as usize;
                let mm_w = ((tw != 0) & (self.bond_between_strands(tw, t) > threshold)) as usize;

                // Should we repurpose one of these to represent strand-scaffold mismatches?
                // These are currently impossible, but could be added in the future.
                // let ts = state.tile_to_s(p);
                // let mm_s = ((ts != 0) & (self.get_energy_ns(t, ts) < threshold)) as usize;

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
        match name {
            "kf" => {
                let kf = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.kf = *kf;
                self.update_system();
                Ok(NeededUpdate::NonZero)
            }
            "strand_concentrations" => {
                let tile_concs = value
                    .downcast_ref::<Array1<f64>>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.strand_concentration.clone_from(tile_concs);
                self.update_system();
                Ok(NeededUpdate::NonZero)
            }
            "temperature" => {
                let temperature = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.change_temperature_to(*temperature);
                Ok(NeededUpdate::NonZero)
            }
            _ => Err(GrowError::NoParameter(name.to_string())),
        }
    }

    fn get_param(&self, name: &str) -> Result<Box<dyn std::any::Any>, crate::base::GrowError> {
        match name {
            "kf" => Ok(Box::new(self.kf)),
            "strand_concentrations" => Ok(Box::new(self.strand_concentration.clone())),
            "energy_bonds" => Ok(Box::new(self.strand_energy_bonds.clone())),
            "temperature" => Ok(Box::new(self.temperature)),
            _ => Err(GrowError::NoParameter(name.to_string())),
        }
    }

    fn system_info(&self) -> String {
        format!(
            "1 dimensional SDC with scaffold of length {} and {} strands",
            self.scaffold.dim().1,
            self.strand_names.len(),
        )
    }
}

impl TileBondInfo for SDC {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.colors[tile_number as usize]
    }

    fn tile_colors(&self) -> &Vec<[u8; 4]> {
        &self.colors
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.strand_names[tile_number as usize].as_str()
    }

    fn tile_names(&self) -> Vec<&str> {
        self.strand_names.iter().map(|s| s.as_str()).collect()
    }

    fn bond_name(&self, bond_number: usize) -> &str {
        self.glue_names[bond_number].as_str()
    }

    fn bond_names(&self) -> Vec<&str> {
        self.glue_names.iter().map(|x| x.as_str()).collect()
    }
}

// impl FromTileSet for SDC {
//     fn from_tileset(tileset: &crate::tileset::TileSet) -> Result<Self, crate::base::RgrowError> {
//         // This gives us parsed names / etc for tiles and glues.  It makes some wrong assumptions (like
//         // that each tile has four edges), but it will do for now.
//         let pc = ProcessedTileSet::from_tileset(tileset)?;

//         // Combine glue strengths (between like numbers) and glue links (between two numbers)
//         let n_glues = pc.glue_strengths.len();
//         let mut glue_links = Array2::zeros((n_glues, n_glues));
//         for (i, strength) in pc.glue_strengths.indexed_iter() {
//             glue_links[(i, i)] = *strength;
//         }
//         for (i, j, strength) in pc.glue_links.iter() {
//             glue_links[(*i, *j)] = *strength;
//         }

//         // Just generate the stuff that will be filled by the model.
//         let energy_bonds = Array2::default((pc.tile_names.len(), pc.tile_names.len()));

//         // We'll default to 64 scaffolds.
//         let (n_scaffolds, scaffold_length) = match tileset.size {
//             Some(Size::Single(x)) => (64, x),
//             Some(Size::Pair((j, x))) => (j, x),
//             None => panic!("Size not specified for SDC model."),
//         };

//         // The tileset input doesn't have a way to specify scaffolds right now.  This generates a buch of 'fake' scaffolds
//         // each with just glues 0 to scaffold_length, which we can at least play around with.
//         let mut scaffold = Array2::<Glue>::zeros((n_scaffolds, scaffold_length));
//         for ((i, j), v) in scaffold.indexed_iter_mut() {
//             *v = j;
//         }

//         let alpha = tileset.alpha.unwrap_or(0.0);

//         // We'll set strand concentrations using stoic and the traditional kTAM Gmc, where
//         // conc = stoic * u0 * exp(-Gmc + alpha) and u0 = 1M, but we really should just have
//         // a way to specify concentrations directly.
//         let strand_concentration = pc
//             .tile_stoics
//             .mapv(|x| x * (-tileset.gmc.unwrap_or(16.0) + alpha).exp());

//         let mut sys = SDC {
//             strand_names: pc.tile_names,
//             glue_names: pc.glue_names,
//             colors: pc.tile_colors,
//             glues: pc.tile_edges,
//             anchor_tiles: Vec::new(),
//             scaffold,
//             // FIXME
//             scaffold_concentration: 0.0,
//             strand_concentration,
//             kf: tileset.kf.unwrap_or(1.0e6),
//             delta_g_matrix: todo!(),
//             entropy_matrix: todo!(),
//             temperature: todo!(),
//             friends_btm: HashMap::new(),
//             strand_energy_bonds: energy_bonds,
//             scaffold_energy_bonds: todo!(),
//         };

//         // This will generate the friends hashamp, as well as the glues, and the energy bonds
//         sys.update_system();

//         Ok(sys)
//     }
// }

// Here is potentially another way to process this, though not done.  Feel free to delete or modify.

use std::hash::Hash;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum RefOrPair {
    Ref(String),
    Pair(String, String),
}

impl From<String> for RefOrPair {
    fn from(r: String) -> Self {
        RefOrPair::Ref(r)
    }
}

impl From<(String, String)> for RefOrPair {
    fn from(p: (String, String)) -> Self {
        RefOrPair::Pair(p.0, p.1)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum SingleOrMultiScaffold {
    Single(Vec<Option<String>>),
    Multi(Vec<Vec<Option<String>>>),
}

impl From<Vec<Option<String>>> for SingleOrMultiScaffold {
    fn from(v: Vec<Option<String>>) -> Self {
        SingleOrMultiScaffold::Single(v)
    }
}

impl From<Vec<Vec<Option<String>>>> for SingleOrMultiScaffold {
    fn from(v: Vec<Vec<Option<String>>>) -> Self {
        SingleOrMultiScaffold::Multi(v)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub struct SDCStrand {
    pub name: Option<String>,
    pub color: Option<String>,
    pub concentration: f64,

    // this may be slightly better, since this way we know that the user wont
    // enter too many glues, eg an array of 5 glues
    pub btm_glue: Option<String>,
    pub left_glue: Option<String>,
    pub right_glue: Option<String>,
}

#[derive(Debug)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum GsOrSeq {
    GS((f64, f64)),
    Seq(String),
}

fn gsorseq_to_gs(gsorseq: &GsOrSeq) -> (f64, f64) {
    match gsorseq {
        GsOrSeq::GS(x) => *x,
        GsOrSeq::Seq(s) => crate::utils::string_dna_dg_ds(s.as_str()),
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub struct SDCParams {
    pub strands: Vec<SDCStrand>,
    pub scaffold: SingleOrMultiScaffold,
    pub scaffold_concentration: f64,
    // Pair with delta G at 37 degrees C and delta S
    pub glue_dg_s: HashMap<RefOrPair, GsOrSeq>,
    pub k_f: f64,
    pub k_n: f64,
    pub k_c: f64,
    pub temperature: f64,
}

/// Triple (x, y, z)
///
/// x: Original input but parsed so that there can be no errors in it (eg. No h**)
/// y: From (eg. h)
/// z: Inverse (eg. h*)
fn self_and_inverse(value: &String) -> (bool, String, String) {
    // Remove all the stars at the end
    let filtered = value.trim_end_matches("*");
    let star_count = value.len() - filtered.len();
    let is_from = star_count % 2 == 0;

    (
        is_from,
        filtered.to_string(),
        format!("{}*", filtered.to_string()),
    )
}

fn get_or_generate(
    map: &mut HashMap<String, usize>,
    count: &mut usize,
    val: Option<String>,
) -> usize {
    // If the user didnt prove a glue, we assume nothign will ever stick
    let str = match val {
        Some(x) => x,
        None => return 0,
    };

    // If we have already generated an id for this glue, then we use it
    let (is_from, fromval, toval) = self_and_inverse(&str);
    let simpl = if is_from { &fromval } else { &toval };
    let res = map.get(simpl);
    if let Some(u) = res {
        return *u;
    }

    map.insert(fromval, *count);
    map.insert(toval, *count + 1);
    *count += 2;

    if is_from {
        *count - 2
    } else {
        *count - 1
    }
}

impl SDC {
    pub fn from_params(params: SDCParams) -> Self {
        let mut glue_name_map: HashMap<String, usize> = HashMap::new();

        // Add one to account for the empty strand
        let strand_count = params.strands.len() + 1;

        let mut strand_names: Vec<String> = Vec::with_capacity(strand_count);
        let mut strand_colors: Vec<[u8; 4]> = Vec::with_capacity(strand_count);
        let mut strand_concentration = Array1::<f64>::zeros(strand_count);
        strand_names.push("null".to_string());
        strand_colors.push([0, 0, 0, 0]);
        strand_concentration[0] = 0.0;

        let mut glues = Array2::<usize>::zeros((strand_count + 1, 3));
        let mut gluenum = 1;

        for (
            id,
            SDCStrand {
                name,
                color,
                concentration,
                left_glue,
                btm_glue,
                right_glue,
            },
        ) in params.strands.into_iter().enumerate()
        {
            // Add the name and the color
            strand_names.push(name.unwrap_or(id.to_string()));

            let color_as_str = color.as_ref().map(|x| x.as_str());
            let color_or_rand = get_color_or_random(&color_as_str).unwrap();
            strand_colors.push(color_or_rand);

            // Add the glues, note that we want to leave idnex (0, _) empty (for the empty tile)
            glues[(id + 1, WEST_GLUE_INDEX)] =
                get_or_generate(&mut glue_name_map, &mut gluenum, left_glue);
            glues[(id + 1, BOTTOM_GLUE_INDEX)] =
                get_or_generate(&mut glue_name_map, &mut gluenum, btm_glue);
            glues[(id + 1, EAST_GLUE_INDEX)] =
                get_or_generate(&mut glue_name_map, &mut gluenum, right_glue);

            // Add the concentrations
            strand_concentration[id + 1] = concentration;
        }

        // Delta G at 37 degrees C
        let mut glue_delta_g = Array2::<f64>::zeros((gluenum, gluenum));
        let mut glue_s = Array2::<f64>::zeros((gluenum, gluenum));

        for (k, gs_or_dna_sequence) in params.glue_dg_s.iter() {
            // here we handle the fact that the user may have input (g, s) or TCGTA...
            let gs = gsorseq_to_gs(gs_or_dna_sequence);

            let (i, j) = match k {
                RefOrPair::Ref(r) => {
                    let (_, base, inverse) = self_and_inverse(r);
                    (base, inverse)
                }
                RefOrPair::Pair(r1, r2) => {
                    let (r1, r1f, r1t) = self_and_inverse(r1);
                    let (r2, r2f, r2t) = self_and_inverse(r2);
                    (if r1 { r1f } else { r1t }, if r2 { r2f } else { r2t })
                }
            };

            // If the user defines the DNA sequence of a glue, but it is never used in any of the
            // strands, then we can ignore it. Also, if the user does use the glue A, but not the
            // glue B, then we can safely ignore the binding strength of A and B, thus
            //
            // (None, None) and (Some, None) are both fine to skip
            //
            // MAYBE it could be better to iterate tglue_dg_s twice, the first time, we just make
            // sure that all strings are inside the glue_name_map, and if they arent, we can add
            // them. The second time around we know that the glues will always be found in the map
            //
            // However, since you cant mutate the strands glues, it shuold be fine to just ignore
            // the glues that do not exist
            let (i, j) = match (glue_name_map.get(&i), glue_name_map.get(&j)) {
                (Some(&x), Some(&y)) => (x, y),
                _ => continue,
            };

            glue_delta_g[[i, j]] = gs.0;
            glue_delta_g[[j, i]] = gs.0;
            glue_s[[i, j]] = gs.1;
            glue_s[[j, i]] = gs.1;
        }

        let scaffold = match params.scaffold {
            SingleOrMultiScaffold::Single(s) => {
                let mut scaffold = Array2::<Glue>::zeros((64, s.len()));
                for (i, maybe_g) in s.iter().enumerate() {
                    if let Some(g) = maybe_g {
                        let x = *glue_name_map
                            .get(g)
                            .expect(format!("ERROR: Glue {} in scaffold not found!", g).as_str());

                        scaffold.index_axis_mut(ndarray::Axis(1), i).fill(x);
                    } else {
                        scaffold.index_axis_mut(ndarray::Axis(1), i).fill(0);
                    }
                }
                scaffold
            }
            SingleOrMultiScaffold::Multi(_m) => todo!(),
        };

        let mut glue_names = vec![String::default(); gluenum];
        for (s, i) in glue_name_map.iter() {
            glue_names[*i] = s.clone();
        }

        SDC::new(
            // TODO: anchor tiles
            vec![],
            strand_names,
            glue_names,
            scaffold,
            strand_concentration,
            params.scaffold_concentration,
            glues,
            strand_colors,
            params.k_f,
            glue_delta_g,
            glue_s,
            params.temperature,
        )
    }
}

/*
*
* EXPERIMENTAL HELPER FUNCIONS
*
* I think that this part maybe could be moved to a different file
* as to not mix implementation of the system with its use
*/

#[cfg_attr(feature = "python", pyclass)]
pub struct AnnealProtocol {
    /// A tuple with initial and final temperatures (in C)
    pub temperatures: (f64, f64),
    /// A tuple with:
    /// 1. How long to hold the initial temperature for before starting the temperature decremenet
    /// 2. How long to hold the final temperature for before finishing the anneal
    pub holds: (f64, f64),
    /// How long to spend in the phase where the temperature is decrementing from the initial to
    /// the final temp
    pub anneal_time: f64,
    /// How long to spend at each temperature
    pub seconds_per_step: f64,
    pub scaffold_count: usize,
}

/// Canvas Arrays, Times, Temperatues
type AnnealOutput = (Vec<Vec<u32>>, Vec<f64>, Vec<f64>);

impl Default for AnnealProtocol {
    fn default() -> Self {
        AnnealProtocol {
            temperatures: (80., 20.),
            holds: (10. * 60., 45. * 60.),
            anneal_time: 3.0 * 60.0 * 60.0,
            seconds_per_step: 2.0,
            scaffold_count: 100,
        }
    }
}

impl AnnealProtocol {
    #[inline(always)]
    fn initial_steps(&self) -> usize {
        (self.holds.0 / self.seconds_per_step).ceil() as usize
    }

    #[inline(always)]
    fn final_steps(&self) -> usize {
        (self.holds.1 / self.seconds_per_step).ceil() as usize
    }

    #[inline(always)]
    fn delta_steps(&self) -> usize {
        (self.anneal_time / self.seconds_per_step).ceil() as usize
    }

    /// Generates two arrays:
    /// (Vec<temperatures>, Vec<times>)
    pub fn generate_arrays(&self) -> (Vec<f64>, Vec<f64>) {
        // See how many steps we wil take during each of the stages
        let steps_init = self.initial_steps();
        let steps_final = self.final_steps();
        let steps_delta = self.delta_steps();

        let mut temps = Vec::<f64>::with_capacity(steps_init + steps_delta + steps_final);
        let mut times = Vec::<f64>::with_capacity(steps_init + steps_delta + steps_final);

        // This assumes that the final temperature is lower
        let temperature_diff = self.temperatures.0 - self.temperatures.1;
        let temperature_delta = temperature_diff / (steps_delta as f64);

        // Initial time in seconds
        let mut current_time = 0.0;
        let mut current_temp = self.temperatures.0;

        (0..steps_init).for_each(|_step_num| {
            current_time += self.seconds_per_step;

            // The temperature doesnt change
            temps.push(current_temp);
            // The time increments by the same delta
            times.push(current_time);
        });

        (0..steps_delta).for_each(|_step_num| {
            current_time += self.seconds_per_step;
            current_temp -= temperature_delta;

            // The temperature doesnt change
            temps.push(current_temp);
            // The time increments by the same delta
            times.push(current_time);
        });

        (0..steps_final).for_each(|_step_num| {
            current_time += self.seconds_per_step;

            // The temperature doesnt change
            temps.push(current_temp);
            // The time increments by the same delta
            times.push(current_time);
        });

        (temps, times)
    }

    // The reason I made this function part of the anneal struct, rather than having this function
    // be part of the SDC is that it will be easier to implement "run_many_systems" and have it be
    // concurrent
    pub fn run_system<St: State>(
        &self,
        mut sdc: SDC,
        mut state: St,
    ) -> Result<AnnealOutput, GrowError> {
        let (tmps, times) = self.generate_arrays();

        let bounds = EvolveBounds::default().for_time(self.seconds_per_step);
        let needed = NeededUpdate::NonZero;
        let mut canvases = Vec::new();

        for tmp in &tmps {
            // Change the temperature
            sdc.temperature = *tmp;
            sdc.update_system();

            crate::system::System::update_state(&sdc, &mut state, &needed);
            crate::system::System::evolve(&sdc, &mut state, bounds)?;
            // FIXME: This is flattening the canvas, so it doesnt work nicely
            // it should be Vec<Vec<_>>, not Vec<_>
            let canvas = state.raw_array().to_slice().unwrap();
            canvases.push(canvas.to_vec())
        }

        Ok((canvases, times, tmps))
    }

    fn default_state(&self, sdc: &SDC) -> Result<StateEnum, GrowError> {
        // There is a better way to do this
        let scaffold_size = sdc.scaffold().len();
        let shape = (self.scaffold_count, scaffold_size);
        let n_tile_types = sdc.strand_names.len();

        StateEnum::empty(
            shape,
            crate::tileset::CanvasType::Square,
            crate::tileset::TrackingType::None,
            n_tile_types,
        )
    }

    fn run_anneal_default_system(&self, sdc: SDC) -> Result<AnnealOutput, GrowError> {
        let state = self.default_state(&sdc)?;
        self.run_system(sdc, state)
    }

    fn run_many_anneals_default_system(
        &self,
        sdcs: Vec<SDC>,
    ) -> Vec<Result<AnnealOutput, GrowError>> {
        sdcs.par_iter()
            .map(|sdc| self.run_anneal_default_system(sdc.clone()))
            .collect()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl AnnealProtocol {
    #[new]
    fn new(
        from_tmp: f64,
        to_tmp: f64,
        initial_hold: f64,
        final_hold: f64,
        delta_time: f64,
        scaffold_count: usize,
        seconds_per_step: f64,
    ) -> Self {
        AnnealProtocol {
            temperatures: (from_tmp, to_tmp),
            seconds_per_step,
            anneal_time: delta_time,
            holds: (initial_hold, final_hold),
            scaffold_count,
        }
    }

    fn run_one_system(&self, sdc: SDC) -> Option<AnnealOutput> {
        self.run_anneal_default_system(sdc).ok()
    }

    fn run_many_systems(&self, sdcs: Vec<SDC>) -> Vec<Option<AnnealOutput>> {
        self.run_many_anneals_default_system(sdcs)
            .into_iter()
            .map(|z| z.ok())
            .collect()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl SDC {
    #[new]
    fn py_new(params: SDCParams) -> Self {
        SDC::from_params(params)
    }

    fn partition(&self) -> f64 {
        self.partition_function_full()
    }

    fn distribution(&self) -> Vec<f64> {
        // Inneficient to run the same function twice, fix this
        let mut probability = self
            .system_states()
            .iter()
            .map(|sys| self.probability_of_state(sys))
            .collect::<Vec<_>>();

        probability.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        probability
    }

    /// Change temperature of the system (degrees C) and update system with that new temperature
    fn set_tmp_c(&mut self, tmp: f64) {
        self.temperature = tmp;
        self.update_system();
    }

    #[getter]
    fn get_scaffold_energy_bonds<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Bound<'py, numpy::PyArray1<f64>> {
        self.fill_energy_array();
        self.scaffold_energy_bonds
            .map(|x| *x.get().unwrap())
            .to_pyarray_bound(py)
    }

    #[getter]
    fn get_strand_energy_bonds<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Bound<'py, numpy::PyArray2<f64>> {
        self.fill_energy_array();
        self.strand_energy_bonds
            .map(|x| *x.get().unwrap())
            .to_pyarray_bound(py)
    }

    #[getter]
    fn get_tile_concs<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.strand_concentration.to_pyarray_bound(py)
    }

    fn get_all_probs(&self) -> Vec<(Vec<u32>, f64, f64)> {
        let systems = self.system_states();
        let mut triples = Vec::new();
        for s in systems {
            let prob = self.probability_of_state(&s);
            let energy = self.boltzman_function(&s);
            triples.push((s, prob, energy));
        }

        triples.sort_unstable_by(|(_, x, _), (_, y, _)| {
            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
        });
        triples
    }

    #[pyo3(name = "partition_function")]
    fn py_partition_function(&self) -> f64 {
        self.partition_function_fast()
    }

    #[pyo3(name = "partition_function_full")]
    fn py_partition_function_full(&self) -> f64 {
        self.partition_function_full()
    }

    #[pyo3(name = "probability_of_state")]
    fn py_probability_of_state(&self, state: Vec<u32>) -> f64 {
        self.probability_of_state(&state)
    }

    #[pyo3(name = "state_g")]
    fn py_state_g(&self, state: Vec<u32>) -> f64 {
        self.g_system(&state)
    }

    #[pyo3(name = "rtval")]
    fn py_rtval(&self) -> f64 {
        self.rtval()
    }

    #[pyo3(name = "log_big_partition_function")]
    fn py_log_big_partition_function(&self) -> f64 {
        self.log_big_partition_function_fast()
    }
}

#[cfg(test)]
mod test_anneal {
    use super::*;

    const ANNEAL: AnnealProtocol = AnnealProtocol {
        temperatures: (88., 28.),
        holds: (10. * 60., 45. * 60.),
        anneal_time: 3.0 * 60.0 * 60.0,
        seconds_per_step: 2.0,
        scaffold_count: 100,
    };

    fn gen_sdc() -> SDC {
        let mut strands = Vec::<SDCStrand>::new();

        // Anchor tile
        strands.push(SDCStrand {
            name: Some("0A0".to_string()),
            color: None,
            concentration: 1e-6,
            btm_glue: Some(String::from("A")),
            left_glue: None,
            right_glue: Some("0e".to_string()),
        });
        strands.push(SDCStrand {
            name: Some("-E-".to_string()),
            color: None,
            concentration: 1e-6,
            btm_glue: Some(String::from("E")),
            left_glue: None,
            right_glue: None,
        });

        for base in "BCD".chars() {
            let (leo, reo): (String, String) = if base == 'C' {
                ("o".to_string(), "e".to_string())
            } else {
                ("e".to_string(), "o".to_string())
            };

            let name = format!("0{}0", base);
            let lg = format!("0{}*", leo);
            let rg = format!("0{}", reo);
            strands.push(SDCStrand {
                name: Some(name),
                color: None,
                concentration: 1e-6,
                btm_glue: Some(String::from(base)),
                left_glue: Some(lg),
                right_glue: Some(rg),
            });

            let name = format!("1{}1", base);
            let lg = format!("1{}*", leo);
            let rg = format!("1{}*", reo);
            strands.push(SDCStrand {
                name: Some(name),
                color: None,
                concentration: 1e-6,
                btm_glue: Some(String::from(base)),
                left_glue: Some(lg),
                right_glue: Some(rg),
            })
        }

        let scaffold = SingleOrMultiScaffold::Single(vec![
            None,
            None,
            Some("A*".to_string()),
            Some("B*".to_string()),
            Some("C*".to_string()),
            Some("D*".to_string()),
            Some("E*".to_string()),
            None,
            None,
        ]);

        let glue_dg_s: HashMap<RefOrPair, GsOrSeq> = HashMap::from(
            [
                ("0e", "GCTGAGAAGAGG"),
                ("1e", "GGATCGGAGATG"),
                ("2e", "GGCTTGGAAAGA"),
                ("3e", "GGCAAGGATTGA"),
                ("4e", "AACAGGGATGTG"),
                ("5e", "AATGGGACATGG"),
                ("6e", "GAACGTTGGTTG"),
                ("7e", "GACGAAGTGTGA"),
                ("0o", "GGTCAGGATGAG"),
                ("1o", "GAACGGAGTTGA"),
                ("2o", "AATGGTGGCATT"),
                ("3o", "GACAAGGGTTGT"),
                ("4o", "TGTTGGGAACAG"),
                ("5o", "GGACTGGTAGTG"),
                ("6o", "GACAGTGTGTGT"),
                ("7o", "GGACGAAAGTGA"),
                ("A", "TCTTTCCAGAGCCTAATTTGCCAG"),
                ("B", "AGCGTCCAATACTGCGGAATCGTC"),
                ("C", "ATAAATATTCATTGAATCCCCCTC"),
                ("D", "AAATGCTTTAAACAGTTCAGAAAA"),
                ("E", "CGAGAATGACCATAAATCAAAAAT"),
            ]
            .map(|(r, g)| (RefOrPair::Ref(r.to_string()), GsOrSeq::Seq(g.to_string()))),
        );

        let sdc_params = SDCParams {
            strands,
            scaffold,
            temperature: 20.0,
            scaffold_concentration: 1e-100,
            glue_dg_s,
            k_f: 1e6,
            k_n: 1e5,
            k_c: 1e4,
        };

        let mut sdc = SDC::from_params(sdc_params);
        sdc.update_system();
        sdc
    }

    #[test]
    fn test_time_and_temp_array() {
        let (tmp, time) = ANNEAL.generate_arrays();

        let mut expected_time = vec![];
        let mut ctime = 2.0;
        loop {
            expected_time.push(ctime);
            ctime += 2.0;
            if ctime > 14100.0 {
                break;
            }
        }
        assert_eq!(time, expected_time);

        (0..300).for_each(|i| {
            let top = tmp[i];
            assert_eq!(top, 88.0);
        });
        let tmps = [
            87.98888683089461,
            87.97777366178921,
            87.96666049268383,
            87.95554732357844,
            87.94443415447304,
            87.93332098536766,
        ];
        (0..6).for_each(|i| {
            let top = tmp[300 + i];
            assert!((tmps[i] - top).abs() < 0.1);
        })
    }

    #[test]
    fn test_run_anneal() {
        let sdc = gen_sdc();
        ANNEAL.run_anneal_default_system(sdc).unwrap();
    }
}

#[cfg(test)]
mod test_sdc_model {
    use crate::assert_all;
    use ndarray::array;
    use num_traits::PrimInt;

    use super::*;
    #[test]
    fn test_update_system() {
        // a lot of the parameters here make no sense, but they wont be used in the tests so it
        // doesnt matter
        let mut sdc = SDC {
            anchor_tiles: Vec::new(),
            strand_names: Vec::new(),
            glue_names: Vec::new(),
            scaffold: Array2::<usize>::zeros((5, 5)),
            strand_concentration: Array1::<f64>::zeros(5),
            scaffold_concentration: 0.0,
            glues: array![
                [0, 0, 0],
                [1, 3, 12],
                [6, 2, 12],
                [31, 3, 45],
                [8, 4, 2],
                [1, 1, 78],
                [4, 4, 1],
            ],
            colors: Vec::new(),
            kf: 0.0,
            friends_btm: HashMap::new(),
            entropy_matrix: array![[1., 2., 3.], [5., 1., 8.], [5., -2., 12.]],
            delta_g_matrix: array![[4., 1., -8.], [6., 1., 14.], [12., 21., -13.,]],
            temperature: 5.,
            strand_energy_bonds: Array2::default((5, 5)),
            scaffold_energy_bonds: Array1::default(5),
        };

        sdc.update_system();

        // THIS TEST WILL NO LONGER PASS, SINCE NOW THE FORMULA IS DIFFERENT
        //
        // TODO: Update test

        // Check that the glue matrix is being generated as expected
        let _expeced_glue_matrix = array![[-1.0, -9., -23.], [-19., -4., -26.], [-13., 31., -73.]];
        // assert_eq!(expeced_glue_matrix, sdc.glue_links);

        // TODO Check that the energy bonds are being generated as expected

        // Check that the friends hashmap is being generated as expected
        let expected_friends = HashMap::from([
            (1, HashSet::from([2])),
            (2, HashSet::from([5])),
            (3, HashSet::from([4, 6])),
            (4, HashSet::from([1, 3])),
        ]);
        assert_eq!(expected_friends, sdc.friends_btm);
    }

    #[test]
    fn test_self_and_inverse() {
        let input = vec!["some*str", "some*str*", "some*str**"];

        let acc = input
            .into_iter()
            .map(|str| self_and_inverse(&str.to_string()))
            .collect::<Vec<(bool, String, String)>>();

        let expected = vec![
            (true, "some*str", "some*str*"),
            (false, "some*str", "some*str*"),
            (true, "some*str", "some*str*"),
        ]
        .iter()
        .map(|(a, b, c)| (*a, b.to_string(), c.to_string()))
        .collect::<Vec<(bool, String, String)>>();

        assert_eq!(acc, expected);
    }

    #[test]
    fn combinations() {
        let mut scaffold = Array2::<usize>::zeros((1, 8));
        scaffold[(0, 2)] = 1;
        scaffold[(0, 3)] = 1;
        scaffold[(0, 4)] = 2;
        scaffold[(0, 5)] = 4;

        let mut sdc = SDC {
            anchor_tiles: Vec::new(),
            strand_names: Vec::new(),
            glue_names: Vec::new(),
            scaffold,
            strand_concentration: Array1::<f64>::zeros(5),
            glues: array![
                [0, 0, 0],
                [1, 3, 12],
                [11, 2, 12],
                [29, 3, 45],
                [8, 4, 2],
                [11, 1, 30],
                [4, 4, 1],
            ],
            scaffold_concentration: 0.0,
            colors: Vec::new(),
            kf: 0.0,
            friends_btm: HashMap::new(),
            entropy_matrix: array![[1., 2., 3.], [5., 1., 8.], [5., -2., 12.]],
            delta_g_matrix: array![[4., 1., -8.], [6., 1., 14.], [12., 21., -13.,]],
            temperature: 50.0,
            strand_energy_bonds: Array2::default((5, 5)),
            scaffold_energy_bonds: Array1::default(5),
        };
        // We need to fill the friends map
        sdc.update_system();

        // 0 <---> Nothing
        //
        // 1 <---> 2
        // 3 <---> 4
        // 5 <---> 6

        assert_eq!(sdc.scaffold(), vec![0, 0, 1, 1, 2, 4, 0, 0]);
        let x = sdc.system_states();

        assert_all!(
            x.contains(&vec![0, 0, 2, 2, 5, 1, 0, 0]),
            x.contains(&vec![0, 0, 2, 2, 5, 1, 0, 0]),
            x.contains(&vec![0, 0, 0, 2, 5, 1, 0, 0]),
            x.contains(&vec![0, 0, 2, 0, 5, 1, 0, 0]),
            x.contains(&vec![0, 0, 2, 2, 0, 1, 0, 0]),
            x.contains(&vec![0, 0, 2, 2, 5, 0, 0, 0]),
            x.contains(&vec![0, 0, 0, 0, 5, 1, 0, 0]),
            x.contains(&vec![0, 0, 0, 0, 5, 1, 0, 0]),
            x.contains(&vec![0, 0, 0, 2, 0, 1, 0, 0]),
            x.contains(&vec![0, 0, 0, 2, 5, 0, 0, 0])
        );

        // Note: One is added to each since the 0 state is not in friends
        //
        //                   vvvvvv friends of 1 (squared since 1 shows up twice)
        //                   vvvvvv          vvvvvv friends of 2
        //                   vvvvvv          vvvvvv     vvvvvv friends of 4
        assert_eq!(x.len(), (1 + 1).pow(2) * (1 + 1) * (2 + 1));
    }

    fn scaffold_for_tests() -> SDC {
        let mut strands = Vec::<SDCStrand>::new();

        // Anchor tile
        strands.push(SDCStrand {
            name: Some("0A0".to_string()),
            color: None,
            concentration: 1e-6,
            btm_glue: Some(String::from("A")),
            left_glue: None,
            right_glue: Some("0e".to_string()),
        });
        strands.push(SDCStrand {
            name: Some("-E-".to_string()),
            color: None,
            concentration: 1e-6,
            btm_glue: Some(String::from("E")),
            left_glue: None,
            right_glue: None,
        });

        for base in "BCD".chars() {
            let (leo, reo): (String, String) = if base == 'C' {
                ("o".to_string(), "e".to_string())
            } else {
                ("e".to_string(), "o".to_string())
            };

            let name = format!("0{}0", base);
            let lg = format!("0{}*", leo);
            let rg = format!("0{}", reo);
            strands.push(SDCStrand {
                name: Some(name),
                color: None,
                concentration: 1e-6,
                btm_glue: Some(String::from(base)),
                left_glue: Some(lg),
                right_glue: Some(rg),
            });

            let name = format!("1{}1", base);
            let lg = format!("1{}*", leo);
            let rg = format!("1{}*", reo);
            strands.push(SDCStrand {
                name: Some(name),
                color: None,
                concentration: 1e-6,
                btm_glue: Some(String::from(base)),
                left_glue: Some(lg),
                right_glue: Some(rg),
            })
        }

        let scaffold = SingleOrMultiScaffold::Single(vec![
            None,
            None,
            Some("A*".to_string()),
            Some("B*".to_string()),
            Some("C*".to_string()),
            Some("D*".to_string()),
            Some("E*".to_string()),
            None,
            None,
        ]);

        let glue_dg_s: HashMap<RefOrPair, GsOrSeq> = HashMap::from(
            [
                ("0e", "GCTGAGAAGAGG"),
                ("1e", "GGATCGGAGATG"),
                ("2e", "GGCTTGGAAAGA"),
                ("3e", "GGCAAGGATTGA"),
                ("4e", "AACAGGGATGTG"),
                ("5e", "AATGGGACATGG"),
                ("6e", "GAACGTTGGTTG"),
                ("7e", "GACGAAGTGTGA"),
                ("0o", "GGTCAGGATGAG"),
                ("1o", "GAACGGAGTTGA"),
                ("2o", "AATGGTGGCATT"),
                ("3o", "GACAAGGGTTGT"),
                ("4o", "TGTTGGGAACAG"),
                ("5o", "GGACTGGTAGTG"),
                ("6o", "GACAGTGTGTGT"),
                ("7o", "GGACGAAAGTGA"),
                ("A", "TCTTTCCAGAGCCTAATTTGCCAG"),
                ("B", "AGCGTCCAATACTGCGGAATCGTC"),
                ("C", "ATAAATATTCATTGAATCCCCCTC"),
                ("D", "AAATGCTTTAAACAGTTCAGAAAA"),
                ("E", "CGAGAATGACCATAAATCAAAAAT"),
            ]
            .map(|(r, g)| (RefOrPair::Ref(r.to_string()), GsOrSeq::Seq(g.to_string()))),
        );

        let sdc_params = SDCParams {
            strands,
            scaffold,
            temperature: 20.0,
            scaffold_concentration: 1e-100,
            glue_dg_s,
            k_f: 1e6,
            k_n: 1e5,
            k_c: 1e4,
        };

        let mut sdc = SDC::from_params(sdc_params);
        sdc.update_system();
        sdc
    }

    #[test]
    fn probablities() {
        let sdc = scaffold_for_tests();
        let scaffold = vec![0, 0, 2, 8, 16, 18, 6, 0, 0];
        assert_eq!(sdc.scaffold(), scaffold);
        let systems = sdc.system_states();

        // A and E have only one strand possible (or empty), and BCD have 2 or empty
        assert_eq!(systems.len(), 2.pow(2) * 3.pow(3));

        let mut probs = systems
            .iter()
            .map(|s| (s.clone(), sdc.probability_of_state(s)))
            .collect::<Vec<_>>();

        probs.sort_by(|(_, p1), (_, p2)| {
            p2.partial_cmp(p1)
                .expect(format!("{} -- {}", p1, p2).as_str())
        });

        // The perfect combination would be all 0's
        // Lets check if that is the case
        // probs.iter().for_each(|(s, p)| {
        //     println!("Probability of {} for {:?}", p, s);
        // });

        assert_eq!(probs[0].0, vec![0, 0, 1, 3, 5, 7, 2, 0, 0]);
    }
}
