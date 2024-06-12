/*
* Important Notes
*
* Given some PointSafe2, in this model, it will represnt two things
* 1. Which of the scaffolds has an event happening
* 2. In which position of the scaffold said event will take place
*
* TODO:
* - There are quite a few expects that need to be handled better
* - _find_monomer_attachment_possibilities_at_point is missing one parameter (because im unsure as
* to what it does)
* */

use std::{
    collections::{HashMap, HashSet},
    usize,
};

use crate::{
    base::{Energy, Glue, Rate, Tile},
    canvas::PointSafe2,
    state::State,
    system::{Event, System, TileBondInfo},
};

use ndarray::prelude::{Array1, Array2};
use serde::{Deserialize, Serialize};

macro_rules! type_alias {
    ($($t:ty => $($i:ident),*);* $(;)?) => {
        $($(type $i = $t;)*)*
    };
}

type_alias!( f64 => Strength, RatePerConc, Conc );

const WEST_GLUE_INDEX: usize = 0;
const BOTTOM_GLUE_INDEX: usize = 1;
const EAST_GLUE_INDEX: usize = 2;

const U0: f64 = 1.0e9;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDC {
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
    pub strands: Array1<Tile>,

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
        self._make_energy_array();

        // I dont think that we need to update the hasmap in this system, as it will never
        // change
    }

    /// Fill the energy_bonds array
    fn _make_energy_array(&mut self) {
        let num_of_strands = self.strands.len();

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

    /// The detachment rate is given by
    ///
    /// TODO: Document the formula here
    pub fn monomer_detachment_rate_at_point<S: State + ?Sized>(
        &self,
        state: &S,
        scaffold_point: PointSafe2,
    ) -> Rate {
        let strand = state.tile_at_point(scaffold_point);

        // There is no strand, thus nothing to be detached
        if strand == 0 {
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
        todo!()
    }

    ///       x y z <- attached strands (potentially empty)
    /// _ _ _ _ _ _ _ _ _ _  <- Scaffold
    ///         ^ point
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
        let friends = match self.friends_btm.get(scaffold_glue) {
            Some(hashset) => hashset,
            None => todo!(),
        };

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
        todo!();
    }

    fn calc_n_tiles<St: State + ?Sized>(&self, state: &St) -> crate::base::NumTiles {
        todo!();
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
            // Empty tile
            0 => self.monomer_detachment_rate_at_point(state, scaffold_coord),
            // Full tile
            _ => self.total_monomer_attachment_rate_at_poin(state, scaffold_coord),
        }
    }

    fn choose_event_at_point<St: State + ?Sized>(
        &self,
        state: &St,
        point: crate::canvas::PointSafe2,
        acc: crate::base::Rate,
    ) -> crate::system::Event {
        // TODO: Missing choose monomer detachment

        match self.choose_monomer_attachment_at_point(state, point, acc) {
            (true, _, event) => event,
            (false, acc, _) => panic!(
                "Rate: {:?}, {:?}, {:?}, {:?}",
                acc,
                point,
                state,
                state.raw_array()
            ),
        }
    }

    fn perform_event<St: State + ?Sized>(
        &self,
        state: &mut St,
        event: &crate::system::Event,
    ) -> &Self {
        match event {
            // Cannot do nothing
            Event::None => panic!("Being asked to perform null event."),

            // Attachments
            Event::MonomerAttachment(point, tile) | Event::MonomerChange(point, tile) => {
                state.set_sa(point, tile)
            }

            Event::PolymerAttachment(v) | Event::PolymerChange(v) => {
                v.iter().for_each(|(point, tile)| state.set_sa(point, tile))
            }

            // Detachments
            Event::MonomerDetachment(point) => state.set_sa(point, &0),
            Event::PolymerDetachment(vector) => {
                for point in vector {
                    state.set_sa(point, &0);
                }
            }
        };

        state.add_events(1);
        state.record_event(event);
        self
    }

    fn seed_locs(&self) -> Vec<(crate::canvas::PointSafe2, Tile)> {
        panic!("This model does not contain seed tiles")
    }

    fn calc_mismatch_locations<St: State + ?Sized>(&self, state: &St) -> Array2<usize> {
        todo!()
    }

    fn set_param(
        &mut self,
        _name: &str,
        _value: Box<dyn std::any::Any>,
    ) -> Result<crate::system::NeededUpdate, crate::base::GrowError> {
        todo!();
    }

    fn get_param(&self, name: &str) -> Result<Box<dyn std::any::Any>, crate::base::GrowError> {
        todo!()
    }

    fn system_info(&self) -> String {
        format!(
            "1 dimensional SDC with scaffold of len {} and {} strands",
            self.scaffold.len(),
            self.strands.len(),
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
