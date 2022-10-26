use crate::base::RgrowError;
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

type GlueNameMap = BiMap<String, Glue>;

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
    RepeatedGlueDef { name: String },
    #[error("Repeated tile definition for {name}.")]
    RepeatedTileName { name: String },
    #[error("No glues found in tileset definition.")]
    NoGlues,
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
            Self::Name(s) => write!(f, "\"{s}\""),
            Self::Num(n) => write!(f, "{n}"),
        }
    }
}

impl core::fmt::Debug for GlueIdent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Name(s) => write!(f, "\"{s}\""),
            Self::Num(n) => write!(f, "{n}"),
        }
    }
}

impl core::fmt::Debug for TileIdent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Name(s) => write!(f, "\"{s}\""),
            Self::Num(n) => write!(f, "{n}"),
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
#[serde(untagged)]
pub enum TileShape {
    Single,
    Horizontal,
    Vertical,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tile {
    pub name: Option<String>,
    pub edges: Vec<GlueIdent>,
    pub stoic: Option<f64>,
    pub color: Option<String>,
    pub shape: Option<TileShape>,
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
            shape: None,
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
            shape: None,
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
    #[serde(default = "Vec::new")]
    pub tiles: Vec<Tile>,
    #[serde(default = "Vec::new")]
    pub bonds: Vec<Bond>,
    #[serde(default = "Vec::new")]
    pub glues: Vec<(GlueIdent, GlueIdent, f64)>,
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
fn block_default() -> Option<usize> {
    None
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
    pub block: Option<usize>,
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

pub trait FromTileSet: Sized {
    fn from_tileset(tileset: &TileSet) -> Result<Self, RgrowError>;
}

pub trait SimFromTileSet {
    fn sim_from_tileset(tileset: &TileSet) -> Result<Box<dyn Simulation>, RgrowError>;
}

impl TileSet {
    pub fn from_json(data: &str) -> serde_json::Result<Self> {
        serde_json::from_str(data)
    }

    pub fn from_yaml(data: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(data)
    }

    pub fn into_simulation(&self) -> Result<Box<dyn Simulation>, RgrowError> {
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
}

/// A processed tile set, suitable for most common models.
pub(crate) struct ProcessedTileSet {
    /// Numbered tile edges.  Single-site tiles only.
    pub(crate) tile_edges: Array2<Glue>,
    /// Tile stoichiometries.
    pub(crate) tile_stoics: Array1<f64>,
    pub(crate) tile_names: Vec<String>,
    pub(crate) tile_colors: Vec<[u8; 4]>,
    pub(crate) glue_names: Vec<String>,
    pub(crate) glue_strengths: Array1<f64>,
    pub(crate) has_duples: bool,

    glue_map: GlueNameMap,
}

impl ProcessedTileSet {
    pub fn from_tileset(tileset: &TileSet) -> Result<Self, ParserError> {
        // Process  glues.
        let mut glue_map = BiMap::new();
        let mut gluestrengthmap = BTreeMap::<Glue, f64>::new();

        let mut gluenum: Glue = 1;

        // We'll deal with zero first, which must be null.
        gluestrengthmap.insert(0, 0.);
        glue_map.insert("0".to_string(), 0);

        // Start with the bond list, which we will take as the more authoritative thing.
        for bond in &tileset.bonds {
            match &bond.name {
                GlueIdent::Name(n) => {
                    glue_map
                        .insert_no_overwrite(n.clone(), gluenum)
                        .map_err(|(l, _r)| ParserError::RepeatedGlueDef { name: l })?;
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

        for tile in &tileset.tiles {
            for name in &tile.edges {
                match &name {
                    GlueIdent::Name(n) => match glue_map.get_by_left(n) {
                        Some(_) => {}

                        None => {
                            glue_map
                                .insert_no_overwrite(n.clone(), gluenum)
                                .map_err(|(l, _r)| ParserError::RepeatedGlueDef { name: l })?;

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

        // Get the highest glue number.
        let highglue = match gluestrengthmap.last_key_value() {
            Some((k, _)) => *k,
            None => Err(ParserError::NoGlues)?,
        };

        let mut glue_strengths = Array1::<f64>::ones(highglue + 1);

        for (j, v) in &gluestrengthmap {
            glue_strengths[*j] = *v;
        }

        let mut tile_names = Vec::with_capacity(tileset.tiles.len() + 1);
        let mut tile_colors = Vec::with_capacity(tileset.tiles.len() + 1);
        let mut tile_stoics = Vec::with_capacity(tileset.tiles.len() + 1);
        let mut tile_edges = Vec::with_capacity((tileset.tiles.len() + 1) * 4);

        // Push the zero state
        tile_names.push("empty".to_string());
        tile_colors.push([0, 0, 0, 0]);
        tile_stoics.push(0.);
        tile_edges.append(&mut vec![0, 0, 0, 0]);

        let mut tile_i = 1;

        let mut v: Vec<Glue> = Vec::new();

        for tile in &tileset.tiles {
            // Ensure the tile name hasn't already been used.
            let tile_name: String = match &tile.name {
                Some(name) => {
                    if tile_names.contains(name) {
                        return Err(ParserError::RepeatedTileName { name: name.clone() });
                    } else {
                        name.clone()
                    }
                }
                None => tile_i.to_string(),
            };

            let tile_stoic = tile.stoic.unwrap_or(1.);

            let tile_color = match &tile.color {
                Some(tc) => *super::colors::COLORS.get(tc.as_str()).unwrap(),
                None => {
                    let mut rng = rand::thread_rng();
                    let ug = rand::distributions::Uniform::new(100u8, 254);
                    [
                        ug.sample(&mut rng),
                        ug.sample(&mut rng),
                        ug.sample(&mut rng),
                        0xffu8,
                    ]
                }
            };

            match &tile.shape.as_ref().unwrap_or(&TileShape::Single) {
                TileShape::Single => {
                    for te in &tile.edges {
                        match te {
                            GlueIdent::Name(n) => v.push(*glue_map.get_by_left(n).unwrap()),
                            GlueIdent::Num(i) => v.push(*i),
                        }
                    }
                    assert!(v.len() == 4);
                    tile_edges.append(&mut v);
                }
                TileShape::Horizontal => todo!(),
                TileShape::Vertical => todo!(),
            }

            tile_names.push(tile_name);
            tile_colors.push(tile_color);
            tile_stoics.push(tile_stoic);

            tile_i += 1;
        }

        Ok(Self {
            tile_edges: Array2::from_shape_vec((tile_i, 4), tile_edges).unwrap(),
            tile_stoics: Array1::from_vec(tile_stoics),
            tile_names,
            tile_colors,
            glue_names: Vec::new(),
            glue_strengths,
            has_duples: false,
            glue_map,
        })
    }

    pub fn tpmap(&self, tp: &TileIdent) -> usize {
        match tp {
            TileIdent::Name(x) => self.tile_names.iter().position(|y| *y == *x).unwrap(),
            TileIdent::Num(x) => *x,
        }
    }

    pub fn gpmap(&self, gp: &GlueIdent) -> usize {
        match gp {
            GlueIdent::Name(x) => *self.glue_map.get_by_left(x).unwrap(),
            GlueIdent::Num(x) => *x,
        }
    }
}
