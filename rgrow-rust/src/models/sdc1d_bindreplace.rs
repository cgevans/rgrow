/// The Bind-Replace model of SDC from the SI of https://www.biorxiv.org/content/10.1101/2025.07.16.664196v1
/// As this model is extremely simple, it also serves as a nice guide to implementing new models.


use core::panic;
use std::collections::HashMap;

use super::sdc1d::{SDCParams, SDCStrand};
use serde::{Deserialize, Serialize};
use crate::{canvas::{PointSafe2, PointSafeHere}, colors::get_color_or_random, models::sdc1d::{SingleOrMultiScaffold, get_or_generate}, state::State, system::{Event, System, TileBondInfo}, units::PerSecond};

#[cfg(feature = "python")]
use numpy::PyArrayMethods;
#[cfg(feature = "python")]
use numpy::ToPyArray;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]   
struct Glue(u64);

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
    matching_tiles_at_site: Vec<Vec<Tile>>,
}

impl TileBondInfo for SDC1DBindReplace   {
    fn tile_color(&self, tile_number: u32) -> [u8; 4] {
        self.strand_colors[tile_number as usize]
    }

    fn tile_colors(&self) -> &Vec<[u8; 4]> {
        &self.strand_colors
    }

    fn tile_name(&self, tile_number: u32) -> &str {
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

    fn event_rate_at_point<St: crate::state::State>(&self, state: &St, p: crate::canvas::PointSafeHere) -> PerSecond {
        if !state.inbounds(p.0) {
            return PerSecond::new(0.0);
        };

        let coord = PointSafe2(p.0);
        match state.tile_at_point(coord) {
            0 => {
                return PerSecond::from(self.matching_tiles_at_site[coord.0.1].len() as f64);
            },
            s => {
                let (glue_w, _, glue_e) = self.strand_glues[s as usize];
                let (glue_to_e, _, _) = self.strand_glues[state.tile_to_e(coord) as usize];
                let (_, _, glue_to_w) = self.strand_glues[state.tile_to_w(coord) as usize];
                let ng = glue_w.matches(glue_to_w) as u8 + glue_e.matches(glue_to_e) as u8;

                let mut n_others = 0.;
                for &possible_replace in &self.matching_tiles_at_site[coord.0.1] {
                    if s == possible_replace.0 {
                        continue;
                    }
                    let (glue_w, _, glue_e) = self.strand_glues[possible_replace.0 as usize];
                    let ng_replace = glue_w.matches(glue_to_w) as u8 + glue_e.matches(glue_to_e) as u8;
                    if ng_replace >= ng {
                        n_others += 1.;
                    }
                }
                return PerSecond::from(n_others);
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
                 for t in &self.matching_tiles_at_site[p.0.1] {
                    mut_acc = mut_acc - PerSecond::from(1.0);
                    if mut_acc.0 <= 0.0 {
                        return (Event::MonomerAttachment(p, t.0), 1.0);
                    }
                 }
                 panic!("Should have found an event to choose at point {:?} with acc {}", p, acc);
            },
            t => {
                let (glue_w, _, glue_e) = self.strand_glues[t.0 as usize];
                let (glue_to_e, _, _) = self.strand_glues[state.tile_to_e(p) as usize];
                let ( _, _, glue_to_w) = self.strand_glues[state.tile_to_w(p) as usize];
                let ng = glue_w.matches(glue_to_w) as u8 + glue_e.matches(glue_to_e) as u8;

                for &possible_replace in &self.matching_tiles_at_site[p.0.1] {
                    if t.0 == possible_replace.0 {
                        continue;
                    }
                    let (glue_w, _, glue_e) = self.strand_glues[possible_replace.0 as usize];
                    let ng_replace = glue_w.matches(glue_to_w) as u8 + glue_e.matches(glue_to_e) as u8;
                    if ng_replace >= ng {
                        mut_acc = mut_acc - PerSecond::from(1.0);
                        if mut_acc.0 <= 0.0 {
                            return (Event::MonomerChange(p, possible_replace.0), 1.0);
                        }
                    }
                }
                panic!("Should have found an event to choose at point {:?} with acc {}", p, acc);
            }
        }
    }

    fn seed_locs(&self) -> Vec<(crate::canvas::PointSafe2, crate::base::Tile)> {
        Vec::default()
    }

    fn calc_mismatch_locations<St: crate::state::State>(&self, state: &St) -> ndarray::Array2<usize> {
        todo!()
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

    pub fn from_params(params: SDCParams) -> Self {
        let mut glue_name_map: HashMap<String, usize> = HashMap::new();
        let strand_count = params.strands.len() + 1; // to account for empty

        let mut strand_names = Vec::with_capacity(strand_count);
        let mut strand_colors = Vec::with_capacity(strand_count);
        let mut glues = Vec::with_capacity(strand_count);
        let mut gluenum = 1;

        strand_names.push("empty".to_string());
        strand_colors.push([0, 0, 0, 0]);
        glues.push((Glue(0), Glue(0), Glue(0)));

        for (
            id,
            SDCStrand {
                name,
                color,
                btm_glue,
                left_glue,
                right_glue,
                ..
            }
        ) in params.strands.into_iter().enumerate() {
            strand_names.push(name.unwrap_or(format!("{}", id)));
            strand_colors.push(get_color_or_random(color.as_deref()).unwrap());

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
                    scaffold.push(Glue(get_or_generate(&mut glue_name_map, &mut gluenum, g) as u64));
                }
                scaffold
            }
            SingleOrMultiScaffold::Multi(_m) => todo!(),
        };

        let mut glue_names = vec![String::default(); gluenum];
        for (sx, i) in glue_name_map.into_iter() {
            glue_names[i] = sx;
        };

        let mut sys = SDC1DBindReplace {
            strand_names,
            strand_colors,
            glue_names,
            scaffold: scaffold.to_vec(),
            strand_glues: glues,
            matching_tiles_at_site: vec![vec![]; scaffold.len()],
        };
        sys.update();
        sys
}

fn update(&mut self) {
    for (i, &scaffold_glue) in self.scaffold.iter().enumerate() {
        let mut matching_tiles = vec![];
        for (tile_num, &(glue_w, glue_b, glue_e)) in self.strand_glues.iter().enumerate() {
            if glue_b.matches(scaffold_glue) {
                matching_tiles.push(Tile(tile_num as u32));
            }
        }
        self.matching_tiles_at_site[i] = matching_tiles;
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
}