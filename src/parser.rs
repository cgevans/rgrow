use crate::system::StaticKTAMCover;

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

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
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

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub enum Direction {
    N,
    E,
    S,
    W
}
#[derive(Serialize, Deserialize, Debug)]
pub struct CoverStrand {
    pub name: Option<String>,
    pub glue: GlueIdent,
    pub dir: Direction,
    pub stoic: f64
}

impl CoverStrand {
    fn to_tile(&self) -> Tile {
        let edges = match self.dir {
            Direction::N => { vec![self.glue.clone(), GlueIdent::Num(0),GlueIdent::Num(0),GlueIdent::Num(0)] }
            Direction::E => { vec![GlueIdent::Num(0), self.glue.clone(), GlueIdent::Num(0),GlueIdent::Num(0)] }
            Direction::S => { vec![GlueIdent::Num(0),GlueIdent::Num(0),self.glue.clone(), GlueIdent::Num(0)] }
            Direction::W => { vec![GlueIdent::Num(0),GlueIdent::Num(0),GlueIdent::Num(0),self.glue.clone()] }
        };

        Tile {
            name: None,
            edges,
            stoic: Some(self.stoic),
            color: None
        }
    }

    fn make_composite(&self, other: &CoverStrand) -> Tile {

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
            color: None
        }
    }
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
    pub cover_strands: Option<Vec<CoverStrand>>
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

    pub fn into_static_ktam_cover(&mut self) -> StaticKTAMCover<QuadTreeState<CanvasSquare, NullStateTracker>> {
        let cs = self.cover_strands.as_ref().unwrap();

        let mut tile_is_cover = Vec::with_capacity(self.tiles.len() + cs.len());
        let mut cover_attach_info = Vec::with_capacity(self.tiles.len() + cs.len());
        let mut composite_detach_info = Vec::with_capacity(self.tiles.len() + cs.len());

        let mut extratiles = Vec::new();

        cover_attach_info.push(Vec::new());
        composite_detach_info.push(Vec::new());
        tile_is_cover.push(system::CoverType::NonCover);

        for _ in 0..self.tiles.len() { 
            tile_is_cover.push(system::CoverType::NonCover); 
            cover_attach_info.push(Vec::new());
            composite_detach_info.push(Vec::new());
         }
        for c in cs { 
            tile_is_cover.push(system::CoverType::Cover);
            composite_detach_info.push(Vec::new());
            cover_attach_info.push(Vec::new());
            extratiles.push(c.to_tile());
        }

        let coverbegin = self.tiles.len() + 1;
        let mut comp = coverbegin + cs.len();

        for i in 0..cs.len() {
            for j in i..cs.len() {
                // Same direction: can't attach at the same place at the same time.
                if cs[i].dir == cs[j].dir { continue }

                assert!(comp == coverbegin + extratiles.len());
                extratiles.push(cs[i].make_composite(&cs[j]));

                cover_attach_info[coverbegin+i].push(system::CoverAttach{ like_tile: (coverbegin+i) as u32, new_tile: comp as u32});
                cover_attach_info[coverbegin+j].push(system::CoverAttach{ like_tile: (coverbegin+j) as u32, new_tile: comp as u32});

                tile_is_cover.push(system::CoverType::Composite);
                composite_detach_info.push(vec![
                    system::CompositeDetach{ like_tile: (coverbegin+i) as u32, new_tile: (coverbegin+j) as u32 },
                    system::CompositeDetach{ like_tile: (coverbegin+j) as u32, new_tile: (coverbegin+i) as u32 }
                ]);

                comp += 1;
            }
        }

        self.tiles.extend(extratiles);

        for tile in self.tiles.iter() {
            println!("{:?}", tile);
        }

        assert!(comp == self.tiles.len()+1);

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

        let inner = StaticKTAM::from_ktam(
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
        );

        StaticKTAMCover { inner, tile_is_cover, cover_attach_info, composite_detach_info }
    }

    pub fn into_static_seeded_ktam<St: state::State>(&self) -> StaticKTAM<St> {
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
