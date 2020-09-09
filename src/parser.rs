use super::base::{Glue, CanvasLength};
use super::*;
use super::system::{StaticATAM, Seed, StaticKTAM, FissionHandling};
use bimap::BiMap;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use ndarray::prelude::*;


#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
enum GlueIdent {
    Name(String),
    Num(Glue),
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
enum ParsedSeed {
    None(),
    Single((CanvasLength, CanvasLength, base::Tile)),
    Multi(Vec<(CanvasLength, CanvasLength, base::Tile)>)
}

#[derive(Serialize, Deserialize, Debug)]
struct Tile {
    name: Option<String>,
    edges: Vec<GlueIdent>,
    stoic: Option<f64>,
    color: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Bond {
    name: GlueIdent,
    strength: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TileSet {
    tiles: Vec<Tile>,
    bonds: Vec<Bond>,
    #[serde(alias="xgrowargs")]
    pub options: Args,
}

fn alpha_default() -> f64 {0.0}
fn gse_default() -> f64 {8.0}
fn gmc_default() -> f64 {16.0}
fn size_default() -> CanvasLength {32}
fn update_rate_default() -> NumEvents {1000}
fn seed_default() -> ParsedSeed { ParsedSeed::None() }
fn fission_default() -> FissionHandling { FissionHandling::KeepLargest }
fn block_default() -> usize {5}

#[derive(Serialize, Deserialize, Debug)]
pub struct Args {
    #[serde(default="gse_default", alias="Gse")]
    gse: f64,
    #[serde(default="gmc_default", alias="Gmc")]
    gmc: f64,
    #[serde(default="alpha_default")]
    alpha: f64,
    #[serde(default="seed_default")]
    seed: ParsedSeed,
    #[serde(default="size_default")]
    pub size: CanvasLength,
    pub tau: Option<f64>,
    pub smax: Option<NumTiles>,
    #[serde(default="update_rate_default")]
    pub update_rate: NumEvents,
    pub kf: Option<f64>,
    #[serde(default="fission_default")]
    pub fission: FissionHandling,
    #[serde(default="block_default")]
    pub block: usize
}

impl TileSet {
    pub fn into_static_seeded_ktam(&self) -> StaticKTAM {
        let (gluemap, gluestrengthmap) = self.number_glues().unwrap();

        let tile_edges = self.tile_edge_process(&gluemap);
        let mut tile_concs = self.tile_stoics();
        tile_concs *= f64::exp(- self.options.gmc - self.options.alpha);

        let mut glue_strength_vec = Vec::<f64>::new();

        let mut i:base::Glue = 0;
        for (j, v) in gluestrengthmap {
            assert!(j==i);
            glue_strength_vec.push(v);
            i+=1;
        } 

        println!("self.options.fission: {:?}", self.options.fission);

        let seed = match &self.options.seed {
            ParsedSeed::Single((y, x, v)) => {Seed::SingleTile{point: (*y,*x), tile: *v}}
            ParsedSeed::None() => {Seed::None()}
            ParsedSeed::Multi(vec) => {let mut hm = HashMap::default();
                hm.extend(vec.iter().map(|(y,x,v)| ((*y,*x),*v))); Seed::MultiTile(hm) }
        };

        StaticKTAM::from_ktam(self.tile_stoics(), tile_edges, Array1::from(glue_strength_vec),
                        self.options.gse, self.options.gmc, Some(self.options.alpha), self.options.kf, Some(seed), Some(self.options.fission),Some(self.tile_names()), Some(self.tile_colors()))

    }

    pub fn into_static_seeded_atam(&self) -> StaticATAM {
        let (gluemap, gluestrengthmap) = self.number_glues().unwrap();

        let tile_edges = self.tile_edge_process(&gluemap);

        let mut tile_concs = self.tile_stoics();
        tile_concs *= f64::exp(- self.options.gmc - self.options.alpha);

        let mut glue_strength_vec = Vec::<f64>::new();

        let mut i:base::Glue = 0;
        for (j, v) in gluestrengthmap {
            assert!(j==i);
            glue_strength_vec.push(v);
            i+=1;
        } 

        let seed = match &self.options.seed {
            ParsedSeed::Single((y, x, v)) => {Seed::SingleTile{point: (*y,*x), tile: *v}}
            ParsedSeed::None() => {Seed::None()}
            ParsedSeed::Multi(vec) => {let mut hm = HashMap::default();
                hm.extend(vec.iter().map(|(y,x,v)| ((*y,*x),*v))); Seed::MultiTile(hm) }
        };

        StaticATAM::new(tile_concs, tile_edges, Array1::from(glue_strength_vec), self.options.tau.unwrap(), Some(seed))
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
                    GlueIdent::Num(i) => v.push(*i)
                }
            }
            assert!(v.len() == 4);
            tile_edges.append(&mut v);
        }
        return Array2::from_shape_vec((tile_edges.len()/4, 4), tile_edges).unwrap()
    }

    pub fn tile_stoics(&self) -> Array1<f64> {
        std::iter::once(0.).chain(self.tiles.iter().map(|x| x.stoic.unwrap_or(1.))).collect()
    }

    pub fn tile_names(&self) -> Vec<String> {
        std::iter::once("empty".to_string()).chain(self.tiles.iter().enumerate().map(|(i, x)| x.name.clone().unwrap_or((i+1).to_string()) )).collect()
    }

    pub fn tile_colors(&self) -> Vec<[u8;4]> {
        let mut tc = Vec::new();

        tc.push([0, 0, 0, 0]);
        let mut rng = rand::thread_rng();
        let ug = rand::distributions::Uniform::new(100u8, 254);
        let ntiles = self.tiles.len()+1;

        for tile in &self.tiles {
            tc.push(match &tile.color {
                Some(tc) => {*super::colors::COLORS.get(tc.as_str()).unwrap()}
                None => {[ug.sample(&mut rng), ug.sample(&mut rng), ug.sample(&mut rng), 0xffu8]}
            });
        }

        tc
    }
}
