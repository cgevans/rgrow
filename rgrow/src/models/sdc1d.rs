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
* - find_monomer_attachment_possibilities_at_point is missing one parameter (because im unsure as
* to what it does)
* - Replace all use of index for glues to WEST_GLUE_INDEX ...
* */

use std::{
    collections::{HashMap, HashSet},
    usize,
};

use crate::{
    base::{Energy, Glue, GrowError, Rate, Tile},
    canvas::{PointSafe2, PointSafeHere},
    state::State,
    system::{Event, NeededUpdate, System, TileBondInfo},
    tileset::{FromTileSet, ProcessedTileSet, Size},
};

use ndarray::prelude::{Array1, Array2};
use serde::{Deserialize, Serialize};

type_alias!( f64 => Strength, RatePerConc, Conc );

const WEST_GLUE_INDEX: usize = 0;
const BOTTOM_GLUE_INDEX: usize = 1;
const EAST_GLUE_INDEX: usize = 2;

const U0: f64 = 1.0e9;

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
    /// Binding strength between two glues
    pub glue_links: Array2<Strength>,
    /// Each strand will be given a color so that it can be easily identified
    /// when illustrated
    pub colors: Vec<[u8; 4]>,
    /// The (de)attachment rates will depend on this constant(for the system) value
    pub kf: RatePerConc,
    /// Constant G_se (TODO: Elaborate)
    pub g_se: Energy,
    pub alpha: Energy,
    /// FIXME: Change this to a vector to avoid hashing time
    ///
    /// Set of tiles that can stick to scaffold gap with a given glue
    pub friends_btm: HashMap<Glue, HashSet<Tile>>,
    /// The energy with which two strands will bond
    ///
    /// This array is indexed as follows. Given strands x and y, where x is to the west of y
    /// (meaning that the east of x forms a bond with the west of y), the energy of said bond
    /// is given by energy_bonds[(x, y)]
    energy_bonds: Array2<Energy>,
}

impl SDC {
    fn update_system(&mut self) {
        // Fill the energy array
        self.fill_energy_array();

        // TODO: Do we also need to update friends here?
    }

    fn polymer_update<S: State + ?Sized>(&self, points: &Vec<PointSafe2>, state: &mut S) {
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

    fn update_monomer_point<S: State + ?Sized>(&self, state: &mut S, scaffold_point: &PointSafe2) {
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
        for strand_f in 0..(num_of_strands as usize) {
            let (f_west_glue, f_east_glue) = {
                let glues = self.glues.row(strand_f);
                (glues[WEST_GLUE_INDEX], glues[EAST_GLUE_INDEX])
            };

            for strand_s in 0..(num_of_strands as usize) {
                let (s_west_glue, s_east_glue) = {
                    let glues = self.glues.row(strand_s);
                    (glues[WEST_GLUE_INDEX], glues[EAST_GLUE_INDEX])
                };

                // Calculate the energy between the two strands

                // Case 1: First strands is to the west of second
                // strand_f    strand_s
                self.energy_bonds[(strand_f, strand_s)] =
                    self.g_se * self.glue_links[(f_east_glue, s_west_glue)];

                // Case 2: First strands is to the east of second
                // strand_s    strand_f
                self.energy_bonds[(strand_s, strand_f)] =
                    self.g_se * self.glue_links[(f_west_glue, s_east_glue)];
            }
        }
    }

    pub fn monomer_detachment_rate_at_point<S: State + ?Sized>(
        &self,
        state: &S,
        scaffold_point: PointSafe2,
    ) -> Rate {
        let strand = state.tile_at_point(scaffold_point);

        let anchor_tile = self.anchor_tiles[(scaffold_point.0).0];

        // If we are trying to detach the anchor tile
        // There is no strand, thus nothing to be detached
        if strand == 0 || anchor_tile.0 == scaffold_point {
            return 0.0;
        }

        let bond_energy = self.bond_energy_of_strand(state, scaffold_point, strand);
        self.kf * (U0 * (-bond_energy + self.alpha).exp())
    }

    pub fn choose_monomer_attachment_at_point<S: State + ?Sized>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: Rate,
    ) -> (bool, Rate, Event) {
        self.find_monomer_attachment_possibilities_at_point(state, acc, point)
    }

    pub fn choose_monomer_detachment_at_point<S: State + ?Sized>(
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
    /// TODO: Add just_calc parameter
    fn find_monomer_attachment_possibilities_at_point<S: State + ?Sized>(
        &self,
        state: &S,
        mut acc: Rate,
        scaffold_coord: PointSafe2,
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
            if acc <= 0.0 {
                return (true, acc, Event::MonomerAttachment(point, strand));
            }
        }

        (false, acc, Event::None)
    }

    fn total_monomer_attachment_rate_at_poin<S: State + ?Sized>(
        &self,
        state: &S,
        scaffold_coord: PointSafe2,
    ) -> f64 {
        // If we set acc = 0, would it not be the case that we just attach to the first tile we can
        // ?
        match self.find_monomer_attachment_possibilities_at_point(state, 0.0, scaffold_coord) {
            (false, acc, _) => -acc,
            _ => panic!(),
        }
    }

    /// Get the sum of the energies of the bonded strands (if any)
    fn bond_energy_of_strand<S: State + ?Sized>(
        &self,
        state: &S,
        scaffold_point: PointSafe2,
        strand: u32,
    ) -> f64 {
        let (w, e) = (
            state.tile_to_w(scaffold_point) as usize,
            state.tile_to_e(scaffold_point) as usize,
        );

        self.energy_bonds[(strand as usize, e)] + self.energy_bonds[(w, strand as usize)]
    }
}

impl System for SDC {
    fn update_after_event<St: State + ?Sized>(&self, state: &mut St, event: &crate::system::Event) {
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

    fn event_rate_at_point<St: State + ?Sized>(
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

    fn choose_event_at_point<St: State + ?Sized>(
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

    // TODO: Array containing locations to "bad connections"
    fn calc_mismatch_locations<St: State + ?Sized>(&self, state: &St) -> Array2<usize> {
        todo!()
    }

    fn set_param(
        &mut self,
        name: &str,
        value: Box<dyn std::any::Any>,
    ) -> Result<crate::system::NeededUpdate, crate::base::GrowError> {
        match name {
            "g_se" => {
                let g_se = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.g_se = *g_se;
                self.update_system();
                Ok(NeededUpdate::NonZero)
            }
            "alpha" => {
                let alpha = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.alpha = *alpha;
                self.update_system();
                Ok(NeededUpdate::NonZero)
            }
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
            _ => Err(GrowError::NoParameter(name.to_string())),
        }
    }

    fn get_param(&self, name: &str) -> Result<Box<dyn std::any::Any>, crate::base::GrowError> {
        match name {
            "g_se" => Ok(Box::new(self.g_se)),
            "alpha" => Ok(Box::new(self.alpha)),
            "kf" => Ok(Box::new(self.kf)),
            "strand_concentrations" => Ok(Box::new(self.strand_concentration.clone())),
            "glue_links" => Ok(Box::new(self.glue_links.clone())),
            _ => Err(GrowError::NoParameter(name.to_string())),
        }
    }

    fn system_info(&self) -> String {
        format!(
            "1 dimensional SDC with scaffold of len {} and {} strands",
            self.scaffold.len(),
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
        for (i, j, strength) in pc.gluelinks.iter() {
            glue_links[(*i, *j)] = *strength;
        }

        // Just generate the stuff that will be filled by the model.
        let energy_bonds = Array2::<f64>::zeros((pc.tile_names.len(), pc.tile_names.len()));
        let friends_btm = HashMap::new();

        // We'll default to 64 scaffolds.
        let (n_scaffolds, scaffold_length) = match tileset.size {
            Some(Size::Single(x)) => (64, x),
            Some(Size::Pair((j, x))) => (j, x),
            None => panic!("Size not specified for SDC model.")
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
        let strand_concentration = pc.tile_stoics.mapv(|x| x * (-tileset.gmc.unwrap_or(16.0) + alpha).exp());

        Ok(SDC {
            strand_names: pc.tile_names,
            glue_names: pc.glue_names,
            glue_links,
            colors: pc.tile_colors,
            glues: pc.tile_edges,
            anchor_tiles: Vec::new(),
            scaffold,
            strand_concentration,
            kf: tileset.kf.unwrap_or(1.0e6),
            g_se: tileset.gse.unwrap_or(5.0),
            alpha,
            friends_btm,
            energy_bonds,
        })
    }
}
