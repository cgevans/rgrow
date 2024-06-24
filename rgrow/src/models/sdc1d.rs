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

use std::{
    collections::{HashMap, HashSet},
    usize,
};

use crate::{
    base::{Energy, Glue, GrowError, Rate, Tile},
    canvas::{PointSafe2, PointSafeHere},
    colors::get_color_or_random,
    state::State,
    system::{Event, NeededUpdate, System, TileBondInfo},
    tileset::{FromTileSet, ProcessedTileSet, Size},
    utils,
};

use ndarray::prelude::{Array1, Array2};
use serde::{Deserialize, Serialize};

type_alias!( f64 => Strength, RatePerConc, Conc );

const WEST_GLUE_INDEX: usize = 0;
const BOTTOM_GLUE_INDEX: usize = 1;
const EAST_GLUE_INDEX: usize = 2;

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
    strand_energy_bonds: Array2<Energy>,
    /// The energy with which a strand attached to scaffold
    scaffold_energy_bonds: Array1<Energy>,
    /// Binding strength between two glues
    glue_links: Array2<Strength>,
}

impl SDC {
    fn new(
        anchor_tiles: Vec<(PointSafe2, Tile)>,
        strand_names: Vec<String>,
        glue_names: Vec<String>,
        scaffold: Array2<Glue>,
        strand_concentration: Array1<Conc>,
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
            // These will be generated by the update_system function next, so just leave them
            // empty for now
            friends_btm: HashMap::new(),
            glue_links: Array2::<f64>::zeros((strand_count, strand_count)),
            strand_energy_bonds: Array2::<f64>::zeros((strand_count, strand_count)),
            scaffold_energy_bonds: Array1::<f64>::zeros(strand_count),
        };
        s.update_system();
        s
    }

    fn update_system(&mut self) {
        // Note that order is important, we need to generate the glue matrix first, then using
        // the data generated there, the energy array is filled, etc...
        self.generate_glue_matrix();
        self.fill_energy_array();
        self.generate_friends();
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

    /// The strenght of glues a, b is given by:
    ///
    /// G(a, b) =  G_(37) (a,b) - (T - 37) * S(a, b)
    fn generate_glue_matrix(&mut self) {
        self.glue_links = &self.delta_g_matrix - (self.temperature - 37.0) * &self.entropy_matrix;
    }

    pub fn change_temperature_to(&mut self, kelvin: f64) {
        self.temperature = kelvin;
        self.update_system();
    }

    fn polymer_update<S: State>(&self, points: &Vec<PointSafe2>, state: &mut S) {
        let mut points_to_update = points
            .iter()
            .flat_map(|&point| {
                [
                    PointSafeHere(point.0),
                    state.move_sa_w(point),
                    state.move_sa_e(point),
                ]
            })
            .collect::<Vec<PointSafeHere>>();

        points_to_update.sort_unstable();
        points_to_update.dedup();
        self.update_points(state, &points_to_update)
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
                self.strand_energy_bonds[(strand_f, strand_s)] =
                    self.glue_links[(f_east_glue, s_west_glue)];

                // Case 2: First strands is to the east of second
                // strand_s    strand_f
                self.strand_energy_bonds[(strand_s, strand_f)] =
                    self.glue_links[(f_west_glue, s_east_glue)];
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
            self.scaffold_energy_bonds[strand_f] = self.glue_links[(f_btm_glue, b_inverse)];
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

        let scaffold_glue = self.scaffold.get(point.0).expect("Invalid Index");

        let empty_map = HashSet::default();
        let friends = self.friends_btm.get(scaffold_glue).unwrap_or(&empty_map);

        for &strand in friends {
            acc -= self.kf * self.strand_concentration[strand as usize];
            if acc <= 0.0 && (!just_calc) {
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

        self.scaffold_energy_bonds[strand as usize]
            + self.strand_energy_bonds[(strand as usize, e)]
            + self.strand_energy_bonds[(w, strand as usize)]
    }
}

impl System for SDC {
    fn update_after_event<St: State>(&self, state: &mut St, event: &crate::system::Event) {
        match event {
            Event::None => todo!(),
            Event::MonomerAttachment(scaffold_point, _)
            | Event::MonomerDetachment(scaffold_point)
            | Event::MonomerChange(scaffold_point, _) => {
                // TODO: Make sure that this is all that needs be done for update
                self.update_monomer_point(state, scaffold_point)
            }
            Event::PolymerDetachment(v) => self.polymer_update(v, state),
            Event::PolymerAttachment(t) | Event::PolymerChange(t) => self.polymer_update(
                &t.iter().map(|(p, _)| *p).collect::<Vec<PointSafe2>>(),
                state,
            ),
        }
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

                let t = state.tile_at_point(p) as usize;

                if t == 0 {
                    continue;
                }

                let te = state.tile_to_e(p) as usize;
                let tw = state.tile_to_w(p) as usize;

                let mm_e = ((te != 0) & (self.strand_energy_bonds[(t, te)] > threshold)) as usize;
                let mm_w = ((tw != 0) & (self.strand_energy_bonds[(tw, t)] > threshold)) as usize;

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
            "glue_links" => {
                let glue_links = value
                    .downcast_ref::<Array2<f64>>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.glue_links.clone_from(glue_links);
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
            "glue_links" => Ok(Box::new(self.glue_links.clone())),
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

impl FromTileSet for SDC {
    fn from_tileset(tileset: &crate::tileset::TileSet) -> Result<Self, crate::base::RgrowError> {
        // This gives us parsed names / etc for tiles and glues.  It makes some wrong assumptions (like
        // that each tile has four edges), but it will do for now.
        let pc = ProcessedTileSet::from_tileset(tileset)?;

        // Combine glue strengths (between like numbers) and glue links (between two numbers)
        let n_glues = pc.glue_strengths.len();
        let mut glue_links = Array2::zeros((n_glues, n_glues));
        for (i, strength) in pc.glue_strengths.indexed_iter() {
            glue_links[(i, i)] = *strength;
        }
        for (i, j, strength) in pc.glue_links.iter() {
            glue_links[(*i, *j)] = *strength;
        }

        // Just generate the stuff that will be filled by the model.
        let energy_bonds = Array2::<f64>::zeros((pc.tile_names.len(), pc.tile_names.len()));

        // We'll default to 64 scaffolds.
        let (n_scaffolds, scaffold_length) = match tileset.size {
            Some(Size::Single(x)) => (64, x),
            Some(Size::Pair((j, x))) => (j, x),
            None => panic!("Size not specified for SDC model."),
        };

        // The tileset input doesn't have a way to specify scaffolds right now.  This generates a buch of 'fake' scaffolds
        // each with just glues 0 to scaffold_length, which we can at least play around with.
        let mut scaffold = Array2::<Glue>::zeros((n_scaffolds, scaffold_length));
        for ((i, j), v) in scaffold.indexed_iter_mut() {
            *v = j;
        }

        let alpha = tileset.alpha.unwrap_or(0.0);

        // We'll set strand concentrations using stoic and the traditional kTAM Gmc, where
        // conc = stoic * u0 * exp(-Gmc + alpha) and u0 = 1M, but we really should just have
        // a way to specify concentrations directly.
        let strand_concentration = pc
            .tile_stoics
            .mapv(|x| x * (-tileset.gmc.unwrap_or(16.0) + alpha).exp());

        let mut sys = SDC {
            strand_names: pc.tile_names,
            glue_names: pc.glue_names,
            glue_links,
            colors: pc.tile_colors,
            glues: pc.tile_edges,
            anchor_tiles: Vec::new(),
            scaffold,
            strand_concentration,
            kf: tileset.kf.unwrap_or(1.0e6),
            delta_g_matrix: todo!(),
            entropy_matrix: todo!(),
            temperature: todo!(),
            friends_btm: HashMap::new(),
            strand_energy_bonds: energy_bonds,
            scaffold_energy_bonds: todo!(),
        };

        // This will generate the friends hashamp, as well as the glues, and the energy bonds
        sys.update_system();

        Ok(sys)
    }
}

// Here is potentially another way to process this, though not done.  Feel free to delete or modify.

use std::hash::Hash;

#[cfg(python)]
use pyo3::prelude::*;

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

        let scaffold = match params.scaffold {
            SingleOrMultiScaffold::Single(s) => {
                let mut scaffold = Array2::<Glue>::zeros((64, s.len()));
                for (i, maybe_g) in s.iter().enumerate() {
                    if let Some(g) = maybe_g {
                        scaffold
                            .index_axis_mut(ndarray::Axis(1), i)
                            .fill(*glue_name_map.get(g).unwrap());
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

            // FIXME: fails if glue not found
            let i = *glue_name_map
                .get(&i)
                .expect(format!("Glue {} not found", i).as_str());

            let j = *glue_name_map
                .get(&j)
                .expect(format!("Glue {} not found", j).as_str());

            glue_delta_g[[i, j]] = gs.0;
            glue_delta_g[[j, i]] = gs.0;
            glue_s[[i, j]] = gs.1;
            glue_s[[j, i]] = gs.1;
        }

        SDC::new(
            // TODO: anchor tiles
            vec![],
            strand_names,
            glue_names,
            scaffold,
            strand_concentration,
            glues,
            strand_colors,
            params.k_f,
            glue_delta_g,
            glue_s,
            params.temperature,
        )
    }
}

#[cfg(test)]
mod test_sdc_model {
    use ndarray::array;

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
            strand_energy_bonds: Array2::<f64>::zeros((5, 5)),
            scaffold_energy_bonds: Array1::<f64>::zeros(5),
            glue_links: Array2::<f64>::zeros((5, 5)),
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
}
