use crate::base::GrowError;
use crate::canvas::{CanvasPeriodic, CanvasSquare, CanvasTube};
use crate::models::atam::ATAM;
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::state::{NullStateTracker, QuadTreeState};

use super::base::{CanvasLength, Glue};
use super::system::FissionHandling;
use super::*;
use base::{NumEvents, NumTiles};
use bimap::BiMap;
use ndarray::prelude::*;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};
use serde_json;
use simulation::Simulation;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::io;
use system::{ChunkHandling, ChunkSize};

use thiserror;

#[derive(thiserror::Error, Debug)]
pub enum ParserError {
    #[error("I/O error: {source}")]
    Io {
        #[source]
        source: io::Error,
    },
    #[error("Inconsistent glue strengths: {name}/{num} has strength {s1} and {s2}.")]
    InconsistentGlueStrength {
        name: GlueIdent,
        num: Glue,
        s1: f64,
        s2: f64,
    },
    #[error("Glue is defined multiple times.")]
    RepeatedGlueDef,
}

#[derive(Serialize, Deserialize, Clone, Eq, PartialEq)]
#[serde(untagged)]
pub enum GlueIdent {
    Name(String),
    Num(Glue),
}

#[derive(Serialize, Deserialize, Clone, Eq, PartialEq)]
#[serde(untagged)]
pub enum TileIdent {
    Name(String),
    Num(usize),
}

impl Display for GlueIdent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Name(s) => write!(f, "\"{}\"", s),
            Self::Num(n) => write!(f, "{}", n),
        }
    }
}

impl core::fmt::Debug for GlueIdent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Name(s) => write!(f, "\"{}\"", s),
            Self::Num(n) => write!(f, "{}", n),
        }
    }
}

impl core::fmt::Debug for TileIdent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Name(s) => write!(f, "\"{}\"", s),
            Self::Num(n) => write!(f, "{}", n),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ParsedSeed {
    None(),
    Single(CanvasLength, CanvasLength, base::Tile),
    Multi(Vec<(CanvasLength, CanvasLength, base::Tile)>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tile {
    pub name: Option<String>,
    pub edges: Vec<GlueIdent>,
    pub stoic: Option<f64>,
    pub color: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub enum Direction {
    N,
    E,
    S,
    W,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoverStrand {
    pub name: Option<String>,
    pub glue: GlueIdent,
    pub dir: Direction,
    pub stoic: f64,
}

impl CoverStrand {
    pub(crate) fn to_tile(&self) -> Tile {
        let edges = match self.dir {
            Direction::N => {
                vec![
                    self.glue.clone(),
                    GlueIdent::Num(0),
                    GlueIdent::Num(0),
                    GlueIdent::Num(0),
                ]
            }
            Direction::E => {
                vec![
                    GlueIdent::Num(0),
                    self.glue.clone(),
                    GlueIdent::Num(0),
                    GlueIdent::Num(0),
                ]
            }
            Direction::S => {
                vec![
                    GlueIdent::Num(0),
                    GlueIdent::Num(0),
                    self.glue.clone(),
                    GlueIdent::Num(0),
                ]
            }
            Direction::W => {
                vec![
                    GlueIdent::Num(0),
                    GlueIdent::Num(0),
                    GlueIdent::Num(0),
                    self.glue.clone(),
                ]
            }
        };

        Tile {
            name: None,
            edges,
            stoic: Some(self.stoic),
            color: None,
        }
    }

    pub(crate) fn make_composite(&self, other: &CoverStrand) -> Tile {
        let es1 = self.to_tile().edges;
        let es2 = other.to_tile().edges;

        let mut edges = Vec::new();
        for (e1, e2) in es1.iter().zip(&es2) {
            if *e1 == GlueIdent::Num(0) {
                edges.push(e2.clone())
            } else {
                edges.push(e1.clone())
            }
        }

        Tile {
            name: None,
            edges,
            stoic: Some(0.),
            color: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Bond {
    pub name: GlueIdent,
    pub strength: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TileSet {
    pub tiles: Vec<Tile>,
    #[serde(default = "Vec::new")]
    pub bonds: Vec<Bond>,
    #[serde(alias = "xgrowargs")]
    pub options: Args,
    pub cover_strands: Option<Vec<CoverStrand>>,
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

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum Size {
    Single(CanvasLength),
    Pair((CanvasLength, CanvasLength)),
}

fn size_default() -> Size {
    Size::Single(32)
}
fn update_rate_default() -> NumEvents {
    1000
}
fn seed_default() -> ParsedSeed {
    ParsedSeed::None()
}
fn fission_default() -> FissionHandling {
    FissionHandling::KeepSeeded
}
fn block_default() -> usize {
    1
}
fn tilepairlist_default() -> Vec<(TileIdent, TileIdent)> {
    Vec::new()
}

#[derive(Serialize, Deserialize, Debug, Clone)]

pub enum CanvasType {
    Square,
    Periodic,
    Tube,
}

fn canvas_type_default() -> CanvasType {
    CanvasType::Periodic
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Model {
    KTAM,
    ATAM,
    OldKTAM,
}

fn threshold_default() -> f64 {
    2.0
}

fn model_default() -> Model {
    Model::KTAM
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Args {
    #[serde(default = "gse_default", alias = "Gse")]
    pub gse: f64,
    #[serde(default = "gmc_default", alias = "Gmc")]
    pub gmc: f64,
    #[serde(default = "alpha_default")]
    pub alpha: f64,
    #[serde(default = "threshold_default")]
    pub threshold: f64,
    #[serde(default = "seed_default")]
    pub seed: ParsedSeed,
    #[serde(default = "size_default")]
    pub size: Size,
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
    pub canvas_type: CanvasType,
    #[serde(default = "tilepairlist_default", alias = "doubletiles")]
    pub hdoubletiles: Vec<(TileIdent, TileIdent)>,
    #[serde(default = "tilepairlist_default")]
    pub vdoubletiles: Vec<(TileIdent, TileIdent)>,
    #[serde(default = "model_default")]
    pub model: Model,
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
            canvas_type: CanvasType::Square,
            hdoubletiles: Vec::new(),
            vdoubletiles: Vec::new(),
            model: model_default(),
            threshold: threshold_default(),
        }
    }
}

pub trait FromTileSet {
    fn from_tileset(tileset: &TileSet) -> Self;
}

pub trait SimFromTileSet {
    fn sim_from_tileset(tileset: &TileSet) -> Result<Box<dyn Simulation>, GrowError>;
}

impl TileSet {
    pub fn from_json(data: &str) -> serde_json::Result<Self> {
        serde_json::from_str(data)
    }

    pub fn from_yaml(data: &str) -> Result<Self, ()> {
        serde_yaml::from_str(data).unwrap_or(Err(()))
    }

    pub fn into_simulation(&self) -> Result<Box<dyn Simulation>, GrowError> {
        match self.options.model {
            Model::KTAM => match self.options.canvas_type {
                CanvasType::Square => {
                    KTAM::<QuadTreeState<CanvasSquare, NullStateTracker>>::sim_from_tileset(self)
                }
                CanvasType::Periodic => {
                    KTAM::<QuadTreeState<CanvasPeriodic, NullStateTracker>>::sim_from_tileset(self)
                }
                CanvasType::Tube => {
                    KTAM::<QuadTreeState<CanvasTube, NullStateTracker>>::sim_from_tileset(self)
                }
            },
            Model::ATAM => match self.options.canvas_type {
                CanvasType::Square => {
                    ATAM::<QuadTreeState<CanvasSquare, NullStateTracker>>::sim_from_tileset(self)
                }
                CanvasType::Periodic => {
                    ATAM::<QuadTreeState<CanvasPeriodic, NullStateTracker>>::sim_from_tileset(self)
                }
                CanvasType::Tube => {
                    ATAM::<QuadTreeState<CanvasTube, NullStateTracker>>::sim_from_tileset(self)
                }
            },
            Model::OldKTAM => match self.options.canvas_type {
                CanvasType::Square => {
                    OldKTAM::<QuadTreeState<CanvasSquare, NullStateTracker>>::sim_from_tileset(self)
                }
                CanvasType::Periodic => {
                    OldKTAM::<QuadTreeState<CanvasPeriodic, NullStateTracker>>::sim_from_tileset(
                        self,
                    )
                }
                CanvasType::Tube => {
                    OldKTAM::<QuadTreeState<CanvasTube, NullStateTracker>>::sim_from_tileset(self)
                }
            },
        }
    }

    pub fn number_glues(&self) -> Result<(BiMap<&str, Glue>, BTreeMap<Glue, f64>), ParserError> {
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
                        .map_err(|(_l, _r)| ParserError::RepeatedGlueDef)?;
                    match gluestrengthmap.get(&gluenum) {
                        Some(s) => {
                            if *s != bond.strength {
                                return Err(ParserError::InconsistentGlueStrength {
                                    name: bond.name.clone(),
                                    num: gluenum,
                                    s1: bond.strength,
                                    s2: bond.strength,
                                });
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
                            return Err(ParserError::InconsistentGlueStrength {
                                name: bond.name.clone(),
                                num: *i,
                                s1: bond.strength,
                                s2: bond.strength,
                            });
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
                            gluemap
                                .insert_no_overwrite(&n, gluenum)
                                .map_err(|(_l, _r)| ParserError::RepeatedGlueDef)?;

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
