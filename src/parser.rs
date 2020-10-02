use super::base::{CanvasLength, Glue};
use super::system::{FissionHandling, Seed, StaticKTAM};
use super::*;
use bimap::BiMap;
use base::{NumEvents, NumTiles};
use canvas::{CanvasPeriodic, CanvasSquare};
use ndarray::prelude::*;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};
use serde_json;
use state::{NullStateTracker, QuadTreeState};
use system::{ChunkHandling, ChunkSize};
use std::collections::{BTreeMap, HashMap};
use std::io;

use thiserror;

#[derive(thiserror::Error, Debug)]
pub enum ParserError {
    #[error("I/O error: {source}")]
    Io {
        #[source]
        source: io::Error,
    },
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum GlueIdent {
    Name(String),
    Num(Glue),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ParsedSeed {
    None(),
    Single(CanvasLength, CanvasLength, base::Tile),
    Multi(Vec<(CanvasLength, CanvasLength, base::Tile)>),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Tile {
    pub name: Option<String>,
    pub edges: Vec<GlueIdent>,
    pub stoic: Option<f64>,
    pub color: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Bond {
    pub name: GlueIdent,
    pub strength: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TileSet {
    pub tiles: Vec<Tile>,
    #[serde(default = "Vec::new")]
    pub bonds: Vec<Bond>,
    #[serde(alias = "xgrowargs")]
    pub options: Args,
}

fn alpha_default() -> f64 {
    0.0
}
fn gse_default() -> f64 {
    8.0
}
fn gmc_default() -> f64 {
    16.0
}
fn size_default() -> CanvasLength {
    32
}
fn update_rate_default() -> NumEvents {
    1000
}
fn seed_default() -> ParsedSeed {
    ParsedSeed::None()
}
fn fission_default() -> FissionHandling {
    FissionHandling::KeepLargest
}
fn block_default() -> usize {
    5
}

#[derive(Serialize, Deserialize, Debug)]

pub enum CanvasType {
    Square,
    Periodic
}

fn canvas_type_default() -> CanvasType {
    CanvasType::Square
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Args {
    #[serde(default = "gse_default", alias = "Gse")]
    pub gse: f64,
    #[serde(default = "gmc_default", alias = "Gmc")]
    pub gmc: f64,
    #[serde(default = "alpha_default")]
    pub alpha: f64,
    #[serde(default = "seed_default")]
    pub seed: ParsedSeed,
    #[serde(default = "size_default")]
    pub size: CanvasLength,
    pub tau: Option<f64>,
    pub smax: Option<NumTiles>,
    #[serde(default = "update_rate_default")]
    pub update_rate: NumEvents,
    pub kf: Option<f64>,
    #[serde(default = "fission_default")]
    pub fission: FissionHandling,
    #[serde(default = "block_default")]
    pub block: usize,
    pub chunk_handling: Option<ChunkHandling>,
    pub chunk_size: Option<ChunkSize>,
    #[serde(default = "canvas_type_default")]
    pub canvas_type: CanvasType
}

impl Default for Args {
    fn default() -> Self {
        Args {
            gse: gse_default(),
            gmc: gmc_default(),
            alpha: alpha_default(),
            seed: seed_default(),
            size: size_default(),
            tau: None,
            smax: None,
            update_rate: update_rate_default(),
            kf: None,
            fission: fission_default(),
            block: block_default(),
            chunk_handling: None,
            chunk_size: None,
            canvas_type: CanvasType::Square
        }
    }
}

impl TileSet {
    pub fn from_json(data: &str) -> serde_json::Result<Self> {
        serde_json::from_str(data)
    }

    pub fn from_yaml(data: &str) -> Result<Self, ()> {
        serde_yaml::from_str(data).unwrap_or(Err(()))
    }

    // pub fn into_system(&self) -> Box<dyn System<dyn State>> {
    //     let (gluemap, gluestrengthmap) = self.number_glues().unwrap();

    //     let tile_edges = self.tile_edge_process(&gluemap);
    //     let mut tile_concs = self.tile_stoics();
    //     tile_concs *= f64::exp(-self.options.gmc + self.options.alpha);

    //     let mut glue_strength_vec = Vec::<f64>::new();

    //     let mut i: base::Glue = 0;
    //     for (j, v) in gluestrengthmap {
    //         assert!(j == i);
    //         glue_strength_vec.push(v);
    //         i += 1;
    //     }

    //     let seed = match &self.options.seed {
    //         ParsedSeed::Single(y, x, v) => Seed::SingleTile {
    //             point: (*y, *x),
    //             tile: *v,
    //         },
    //         ParsedSeed::None() => Seed::None(),
    //         ParsedSeed::Multi(vec) => {
    //             let mut hm = HashMap::default();
    //             hm.extend(vec.iter().map(|(y, x, v)| ((*y, *x), *v)));
    //             Seed::MultiTile(hm)
    //         }
    //     };

    //     Box::new(StaticKTAM::<QuadTreeState<CanvasPeriodic, NullStateTracker>>::from_ktam(
    //         self.tile_stoics(),
    //         tile_edges,
    //         Array1::from(glue_strength_vec),
    //         self.options.gse,
    //         self.options.gmc,
    //         Some(self.options.alpha),
    //         self.options.kf,
    //         Some(seed),
    //         Some(self.options.fission),
    //         self.options.chunk_handling,
    //         self.options.chunk_size,
    //         Some(self.tile_names()),
    //         Some(self.tile_colors()),
    //     ))

    // }


    pub fn into_static_seeded_ktam(&self) -> StaticKTAM<QuadTreeState<CanvasSquare, NullStateTracker>> {
        let (gluemap, gluestrengthmap) = self.number_glues().unwrap();

        let tile_edges = self.tile_edge_process(&gluemap);
        let mut tile_concs = self.tile_stoics();
        tile_concs *= f64::exp(-self.options.gmc + self.options.alpha);

        let mut glue_strength_vec = Vec::<f64>::new();

        let mut i: base::Glue = 0;
        for (j, v) in gluestrengthmap {
            assert!(j == i);
            glue_strength_vec.push(v);
            i += 1;
        }

        let seed = match &self.options.seed {
            ParsedSeed::Single(y, x, v) => Seed::SingleTile {
                point: (*y, *x),
                tile: *v,
            },
            ParsedSeed::None() => Seed::None(),
            ParsedSeed::Multi(vec) => {
                let mut hm = HashMap::default();
                hm.extend(vec.iter().map(|(y, x, v)| ((*y, *x), *v)));
                Seed::MultiTile(hm)
            }
        };

        StaticKTAM::from_ktam(
            self.tile_stoics(),
            tile_edges,
            Array1::from(glue_strength_vec),
            self.options.gse,
            self.options.gmc,
            Some(self.options.alpha),
            self.options.kf,
            Some(seed),
            Some(self.options.fission),
            self.options.chunk_handling,
            self.options.chunk_size,
            Some(self.tile_names()),
            Some(self.tile_colors()),
        )
    }

    pub fn into_static_seeded_ktam_p(&self) -> StaticKTAM<QuadTreeState<CanvasPeriodic, NullStateTracker>> {
        let (gluemap, gluestrengthmap) = self.number_glues().unwrap();

        let tile_edges = self.tile_edge_process(&gluemap);
        let mut tile_concs = self.tile_stoics();
        tile_concs *= f64::exp(-self.options.gmc + self.options.alpha);

        let mut glue_strength_vec = Vec::<f64>::new();

        let mut i: base::Glue = 0;
        for (j, v) in gluestrengthmap {
            assert!(j == i);
            glue_strength_vec.push(v);
            i += 1;
        }

        let seed = match &self.options.seed {
            ParsedSeed::Single(y, x, v) => Seed::SingleTile {
                point: (*y, *x),
                tile: *v,
            },
            ParsedSeed::None() => Seed::None(),
            ParsedSeed::Multi(vec) => {
                let mut hm = HashMap::default();
                hm.extend(vec.iter().map(|(y, x, v)| ((*y, *x), *v)));
                Seed::MultiTile(hm)
            }
        };

        StaticKTAM::from_ktam(
            self.tile_stoics(),
            tile_edges,
            Array1::from(glue_strength_vec),
            self.options.gse,
            self.options.gmc,
            Some(self.options.alpha),
            self.options.kf,
            Some(seed),
            Some(self.options.fission),
            self.options.chunk_handling,
            self.options.chunk_size,
            Some(self.tile_names()),
            Some(self.tile_colors()),
        )
    }

    pub fn number_glues(&self) -> Result<(BiMap<&str, Glue>, BTreeMap<Glue, f64>), ()> {
        let mut gluemap = BiMap::new();
        let mut gluestrengthmap = BTreeMap::<Glue, f64>::new();

        let mut gluenum: Glue = 1;

        // We'll deal with zero first, which must be null.
        gluestrengthmap.insert(0, 0.);
        gluemap.insert("0", 0);

        // Start with the bond list, which we will take as the more authoritative thing.
        for bond in &self.bonds {
            match &bond.name {
                GlueIdent::Name(n) => {
                    gluemap
                        .insert_no_overwrite(&n, gluenum)
                        .expect("Glue already here");
                    match gluestrengthmap.get(&gluenum) {
                        Some(s) => {
                            if *s != bond.strength {
                                return Err(());
                            }
                        }

                        None => {
                            gluestrengthmap.insert(gluenum, bond.strength);
                        }
                    }
                    gluenum += 1;
                }
                GlueIdent::Num(i) => match gluestrengthmap.get(i) {
                    Some(s) => {
                        if *s != bond.strength {
                            return Err(());
                        }
                    }

                    None => {
                        gluestrengthmap.insert(*i, bond.strength);
                    }
                },
            }
        }

        for tile in &self.tiles {
            for name in &tile.edges {
                match &name {
                    GlueIdent::Name(n) => match gluemap.get_by_left(&n.as_str()) {
                        Some(_) => {}

                        None => {
                            gluemap.insert_no_overwrite(&n, gluenum).unwrap();

                            match gluestrengthmap.get(&gluenum) {
                                Some(_) => {}

                                None => {
                                    gluestrengthmap.insert(gluenum, 1.0);
                                }
                            }
                            gluenum += 1;
                        }
                    },
                    GlueIdent::Num(i) => match gluestrengthmap.get(i) {
                        Some(_) => {}

                        None => {
                            gluestrengthmap.insert(*i, 1.0);
                        }
                    },
                }
            }
        }

        Ok((gluemap, gluestrengthmap))
    }

    pub fn tile_edge_process(&self, gluemap: &BiMap<&str, Glue>) -> Array2<Glue> {
        let mut tile_edges: Vec<Glue> = Vec::new();

        tile_edges.append(&mut vec![0, 0, 0, 0]);

        let mut v: Vec<Glue> = Vec::new();
        for tile in &self.tiles {
            for te in &tile.edges {
                match te {
                    GlueIdent::Name(n) => v.push(*gluemap.get_by_left(&n.as_str()).unwrap()),
                    GlueIdent::Num(i) => v.push(*i),
                }
            }
            assert!(v.len() == 4);
            tile_edges.append(&mut v);
        }
        return Array2::from_shape_vec((tile_edges.len() / 4, 4), tile_edges).unwrap();
    }

    pub fn tile_stoics(&self) -> Array1<f64> {
        std::iter::once(0.)
            .chain(self.tiles.iter().map(|x| x.stoic.unwrap_or(1.)))
            .collect()
    }

    pub fn tile_names(&self) -> Vec<String> {
        std::iter::once("empty".to_string())
            .chain(
                self.tiles
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x.name.clone().unwrap_or((i + 1).to_string())),
            )
            .collect()
    }

    pub fn tile_colors(&self) -> Vec<[u8; 4]> {
        let mut tc = Vec::new();

        tc.push([0, 0, 0, 0]);
        let mut rng = rand::thread_rng();
        let ug = rand::distributions::Uniform::new(100u8, 254);

        for tile in &self.tiles {
            tc.push(match &tile.color {
                Some(tc) => *super::colors::COLORS.get(tc.as_str()).unwrap(),
                None => [
                    ug.sample(&mut rng),
                    ug.sample(&mut rng),
                    ug.sample(&mut rng),
                    0xffu8,
                ],
            });
        }

        tc
    }
}
