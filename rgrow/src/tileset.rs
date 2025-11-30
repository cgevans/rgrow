use crate::base::{GlueIdent, RgrowError, StringConvError, TileIdent};
use crate::colors::get_color_or_random;

use crate::models::atam::ATAM;
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::system::{DynSystem, EvolveBounds};

use self::state::StateEnum;
use self::system::{NeededUpdate, SystemEnum};

use super::base::{CanvasLength, Glue};
use super::system::FissionHandling;
use super::*;
use anyhow::Context;
use base::{NumEvents, NumTiles};
use bimap::BiMap;
use core::fmt;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::io::{self, Read};

use std::path::Path;
use system::{ChunkHandling, ChunkSize};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::IntoPyObjectExt;

use thiserror;

type GlueNameMap = BiMap<String, Glue>;

pub const MODEL_DEFAULT: Model = Model::KTAM;
pub const GMC_DEFAULT: f64 = 16.0;
pub const GSE_DEFAULT: f64 = 8.1;
pub const CANVAS_TYPE_DEFAULT: CanvasType = CanvasType::Square;
pub const SIZE_DEFAULT: Size = Size::Single(64);

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
    #[error(transparent)]
    ColorError(#[from] colors::ColorError),
    #[error("Tile {name} has {num} edges, but is a {shape} tile.")]
    WrongNumberOfEdges {
        name: String,
        num: usize,
        shape: TileShape,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub enum Seed {
    Single(CanvasLength, CanvasLength, TileIdent),
    Multi(Vec<(CanvasLength, CanvasLength, TileIdent)>),
}

impl Display for Seed {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(x, y, tile) => write!(f, "Single({x},{y},{tile})"),
            Self::Multi(v) => {
                write!(f, "Multi(")?;
                for (x, y, tile) in v {
                    write!(f, "({x},{y},{tile})")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for Seed {
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Seed::Single(x, y, z) => (x, y, z).into_bound_py_any(py),
            Seed::Multi(v) => v.into_bound_py_any(py),
        }
    }

    type Target = pyo3::PyAny; // the Python type
    type Output = pyo3::Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TileShape {
    #[serde(alias = "single", alias = "s", alias = "S")]
    Single,
    #[serde(alias = "horizontal", alias = "h", alias = "H")]
    Horizontal,
    #[serde(alias = "vertical", alias = "v", alias = "V")]
    Vertical,
}

#[cfg(feature = "python")]
impl FromPyObject<'_> for TileShape {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let s = ob.extract::<String>()?;
        match s.to_lowercase().as_str() {
            "single" | "s" => Ok(Self::Single),
            "horizontal" | "h" => Ok(Self::Horizontal),
            "vertical" | "v" => Ok(Self::Vertical),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown tile shape {s}"
            ))),
        }
    }
}

impl Display for TileShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single => write!(f, "single"),
            Self::Horizontal => write!(f, "horizontal double"),
            Self::Vertical => write!(f, "vertical double"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub struct Tile {
    /// The name of the tile.  If unset, the eventual
    /// number of the tile will be used.
    pub name: Option<String>,
    /// Glues on each edge of the tile, arranged clockwise
    /// from the north (or north-west) edge.  Should be either
    /// four (for a single tile) or six elements.  A 0 is interpreted
    /// as being a null glue.
    pub edges: Vec<GlueIdent>,
    /// Stoichiometric ratio to the default concentration.  Defaults to 1.0.
    pub stoic: Option<f64>,
    /// Color of the tile, as a string.  Either an X11-like name (see colors.rs),
    /// or a #/@ hex string of three u8 values, as often used elsewhere.
    pub color: Option<String>,
    /// The tile shape: whether the tile is a single, horizontal duple, or
    /// vertical duple.
    pub shape: Option<TileShape>,
}

impl Display for Tile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tile {{ ")?;
        if let Some(name) = &self.name {
            write!(f, "name: \"{name}\", ")?;
        }
        write!(f, "edges: [")?;
        for edge in &self.edges {
            write!(f, "{edge}, ")?;
        }
        write!(f, "]")?;
        if let Some(stoic) = self.stoic {
            write!(f, ", stoic: {stoic}")?;
        }
        if let Some(color) = &self.color {
            write!(f, ", color: {color}, ")?;
        }
        if let Some(shape) = &self.shape {
            write!(f, ", shape: {shape}, ")?;
        }
        write!(f, "}}")
    }
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow"))]
pub enum Direction {
    N,
    E,
    S,
    W,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub struct Bond {
    pub name: GlueIdent,
    pub strength: f64,
}

// The purpose of this recursive version of TileSet is to allow serde to parse
// files with an "option" field for backwards compatibility.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
struct SerdeTileSet {
    #[serde(default = "Vec::new")]
    pub(self) tiles: Vec<Tile>,
    #[serde(default = "Vec::new")]
    pub(self) bonds: Vec<Bond>,
    #[serde(default = "Vec::new")]
    pub(self) glues: Vec<(GlueIdent, GlueIdent, f64)>,
    #[serde(alias = "Gse")]
    pub(self) gse: Option<f64>,
    #[serde(alias = "Gmc")]
    pub(self) gmc: Option<f64>,
    #[serde(alias = "α")]
    pub(self) alpha: Option<f64>,
    pub(self) threshold: Option<f64>,
    pub(self) seed: Option<Seed>,
    pub(self) size: Option<Size>,
    pub(self) tau: Option<f64>,
    pub(self) smax: Option<NumTiles>,
    pub(self) update_rate: Option<NumEvents>,
    #[serde(alias = "k_f")]
    pub(self) kf: Option<f64>,
    pub(self) fission: Option<FissionHandling>,
    pub(self) block: Option<usize>,
    pub(self) chunk_handling: Option<ChunkHandling>,
    pub(self) chunk_size: Option<ChunkSize>,
    pub(self) canvas_type: Option<CanvasType>,
    pub(self) tracking: Option<TrackingType>,
    #[serde(alias = "doubletiles")]
    pub(self) hdoubletiles: Option<Vec<(TileIdent, TileIdent)>>,
    pub(self) vdoubletiles: Option<Vec<(TileIdent, TileIdent)>>,
    pub(self) model: Option<Model>,
    #[serde(default)]
    pub(self) start_paused: bool,
    #[serde(alias = "xgrowargs", alias = "params")]
    pub(self) options: Option<Box<SerdeTileSet>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
#[serde(from = "SerdeTileSet")]
pub struct TileSet {
    #[serde(default = "Vec::new")]
    pub tiles: Vec<Tile>,
    #[serde(default = "Vec::new")]
    pub bonds: Vec<Bond>,
    #[serde(default = "Vec::new")]
    pub glues: Vec<(GlueIdent, GlueIdent, f64)>,
    pub gse: Option<f64>,
    pub gmc: Option<f64>,
    pub alpha: Option<f64>,
    pub threshold: Option<f64>,
    pub seed: Option<Seed>,
    pub size: Option<Size>,
    pub tau: Option<f64>,
    pub smax: Option<NumTiles>,
    pub update_rate: Option<NumEvents>,
    pub kf: Option<f64>,
    pub fission: Option<FissionHandling>,
    pub block: Option<usize>,
    pub chunk_handling: Option<ChunkHandling>,
    pub chunk_size: Option<ChunkSize>,
    pub canvas_type: Option<CanvasType>,
    pub tracking: Option<TrackingType>,
    pub hdoubletiles: Option<Vec<(TileIdent, TileIdent)>>,
    pub vdoubletiles: Option<Vec<(TileIdent, TileIdent)>>,
    pub model: Option<Model>,
    #[serde(default)]
    pub start_paused: bool,
}

impl From<SerdeTileSet> for TileSet {
    fn from(serde_tile_set: SerdeTileSet) -> Self {
        let SerdeTileSet {
            tiles,
            bonds,
            glues,
            gse,
            gmc,
            alpha,
            threshold,
            seed,
            size,
            tau,
            smax,
            update_rate,
            kf,
            fission,
            block,
            chunk_handling,
            chunk_size,
            canvas_type,
            tracking,
            hdoubletiles,
            vdoubletiles,
            model,
            start_paused,
            options,
        } = serde_tile_set;

        let mut tile_set = TileSet {
            tiles,
            bonds,
            glues,
            gse,
            gmc,
            alpha,
            threshold,
            seed,
            size,
            tau,
            smax,
            update_rate,
            kf,
            fission,
            block,
            chunk_handling,
            chunk_size,
            canvas_type,
            tracking,
            hdoubletiles,
            vdoubletiles,
            model,
            start_paused,
        };

        if let Some(options) = options {
            tile_set.tiles.extend(options.tiles);
            tile_set.bonds.extend(options.bonds);
            tile_set.glues.extend(options.glues);
            tile_set.gse = options.gse.or(tile_set.gse);
            tile_set.gmc = options.gmc.or(tile_set.gmc);
            tile_set.alpha = options.alpha.or(tile_set.alpha);
            tile_set.threshold = options.threshold.or(tile_set.threshold);
            tile_set.seed = options.seed.or(tile_set.seed);
            tile_set.size = options.size.or(tile_set.size);
            tile_set.tau = options.tau.or(tile_set.tau);
            tile_set.smax = options.smax.or(tile_set.smax);
            tile_set.update_rate = options.update_rate.or(tile_set.update_rate);
            tile_set.kf = options.kf.or(tile_set.kf);
            tile_set.fission = options.fission.or(tile_set.fission);
            tile_set.block = options.block.or(tile_set.block);
            tile_set.chunk_handling = options.chunk_handling.or(tile_set.chunk_handling);
            tile_set.chunk_size = options.chunk_size.or(tile_set.chunk_size);
            tile_set.canvas_type = options.canvas_type.or(tile_set.canvas_type);
            tile_set.tracking = options.tracking.or(tile_set.tracking);
            tile_set.hdoubletiles = options.hdoubletiles.or(tile_set.hdoubletiles);
            tile_set.vdoubletiles = options.vdoubletiles.or(tile_set.vdoubletiles);
            tile_set.model = options.model.or(tile_set.model);
            tile_set.start_paused = options.start_paused || tile_set.start_paused;
        }

        tile_set
    }
}

impl Display for TileSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "TileSet(")?;
        writeln!(f, "    tiles=[")?;
        for tile in &self.tiles {
            writeln!(f, "        {tile},")?;
        }
        writeln!(f, "    ],")?;
        if !self.bonds.is_empty() {
            writeln!(f, "    bonds=[")?;
            for bond in &self.bonds {
                writeln!(f, "        ({}, {}),", bond.name, bond.strength)?;
            }
            writeln!(f, "    ],")?;
        };
        if !self.glues.is_empty() {
            writeln!(f, "    glues=[")?;
            for (a, b, s) in &self.glues {
                writeln!(f, "        ({a}, {b}, {s}),")?;
            }
            writeln!(f, "    ],")?;
        };
        writeln!(f, "    options=[")?;
        if let Some(x) = self.gse {
            writeln!(f, "        Gse: {x}")?;
        }
        if let Some(x) = self.gmc {
            writeln!(f, "        Gmc: {x}")?;
        }
        if let Some(x) = self.alpha {
            writeln!(f, "        alpha: {x}")?;
        }
        if let Some(x) = &self.seed {
            writeln!(f, "        seed: {x}")?;
        }
        if let Some(x) = self.size {
            writeln!(f, "        size: {x}")?;
        }
        writeln!(f, "        tau: {:?}", self.tau)?;
        writeln!(f, "        smax: {:?}", self.smax)?;
        if let Some(x) = self.update_rate {
            writeln!(f, "        update_rate: {x}")?;
        }
        writeln!(f, "        kf: {:?}", self.kf)?;
        writeln!(f, "        fission: {:?}", self.fission)?;
        writeln!(f, "        block: {:?}", self.block)?;
        writeln!(f, "        chunk_handling: {:?}", self.chunk_handling)?;
        writeln!(f, "        chunk_size: {:?}", self.chunk_size)?;
        writeln!(f, "        canvas_type: {:?}", self.canvas_type)?;
        writeln!(f, "        tracking: {:?}", self.tracking)?;
        writeln!(f, "        hdoubletiles: {:?}", self.hdoubletiles)?;
        writeln!(f, "        vdoubletiles: {:?}", self.vdoubletiles)?;
        writeln!(f, "        model: {:?}", self.model)?;
        writeln!(f, "        threshold: {:?}", self.threshold)?;
        write!(f, "    ]\n)\n")?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(untagged)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub enum Size {
    Single(CanvasLength),
    Pair((CanvasLength, CanvasLength)),
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for Size {
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Size::Single(x) => x.into_bound_py_any(py),
            Size::Pair(p) => p.into_bound_py_any(py),
        }
    }

    type Target = pyo3::PyAny; // the Python type
    type Output = pyo3::Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy)]

pub enum CanvasType {
    #[serde(alias = "square")]
    Square,
    #[serde(alias = "periodic")]
    Periodic,
    #[serde(alias = "tube")]
    Tube,
    #[serde(alias = "tube-diagonals")]
    TubeDiagonals,
}

#[cfg(feature = "python")]
impl FromPyObject<'_> for CanvasType {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let s: &str = ob.extract()?;
        CanvasType::try_from(s)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for CanvasType {
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            CanvasType::Square => "square".into_bound_py_any(py),
            CanvasType::Periodic => "periodic".into_bound_py_any(py),
            CanvasType::Tube => "tube".into_bound_py_any(py),
            CanvasType::TubeDiagonals => "tube-diagonals".into_bound_py_any(py),
        }
    }

    type Target = pyo3::PyAny; // the Python type
    type Output = pyo3::Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow", eq, eq_int))]
pub enum TrackingType {
    None,
    Order,
    LastAttachTime,
    PrintEvent,
    Movie,
}

impl TryFrom<&str> for CanvasType {
    type Error = StringConvError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "square" => Ok(CanvasType::Square),
            "periodic" => Ok(CanvasType::Periodic),
            "tube" => Ok(CanvasType::Tube),
            "tube-diagonals" => Ok(CanvasType::TubeDiagonals),
            _ => Err(StringConvError(format!("Unknown canvas type {value}.  Valid options are \"square\", \"periodic\", and \"tube\"."))),
        }
    }
}

impl TryFrom<&str> for TrackingType {
    type Error = StringConvError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "none" => Ok(TrackingType::None),
            "order" => Ok(TrackingType::Order),
            "lastattachtime" => Ok(TrackingType::LastAttachTime),
            "printevent" => Ok(TrackingType::PrintEvent),
            "movie" => Ok(TrackingType::Movie),
            _ => Err(StringConvError(format!(
                "Unknown tracking type {value}.  Valid options are \"none\", \"order\", \"lastattachtime\", \"printevent\", \"movie\"."
            ))),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Model {
    #[serde(alias = "kTAM", alias = "ktam")]
    KTAM,
    #[serde(alias = "aTAM", alias = "atam")]
    ATAM,
    #[serde(alias = "OldkTAM", alias = "oldktam")]
    OldKTAM,
    #[serde(alias = "SDC1D", alias = "sdc1d")]
    SDC,
}

use std::convert::TryFrom;

impl TryFrom<&str> for Model {
    type Error = StringConvError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "ktam" => Ok(Model::KTAM),
            "atam" => Ok(Model::ATAM),
            "oldktam" => Ok(Model::OldKTAM),
            "sdc1d" => Ok(Model::SDC),
            _ => Err(StringConvError(format!(
                "Unknown model {s}. Valid options are kTAM, aTAM, and oldkTAM."
            ))),
        }
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Size::Single(cl) => write!(f, "{cl}"),
            Size::Pair((cl1, cl2)) => write!(f, "({cl1}, {cl2})"),
        }
    }
}

impl TileSet {
    pub fn from_json(data: &str) -> serde_json::Result<Self> {
        serde_json::from_str(data)
    }

    pub fn from_yaml(data: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(data)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error> {
        let mut file = std::fs::File::open(path)?;

        let mut s = String::new();
        file.read_to_string(&mut s)?;

        let res: Result<TileSet, _> = serde_yaml::from_str(&s);

        if let Ok(ts) = res {
            return Ok(ts);
        }

        let res2 = parser_xgrow::parse_xgrow_string(&s);

        if let Ok(ts) = res2 {
            return Ok(ts);
        }

        // We've failed on both.  Do we look like an xgrow file?
        if s.contains("tile edges={") {
            res2.context("Failed to parse xgrow file")
        } else {
            res.context("Failed to parse yaml file")
        }
    }

    pub fn create_system(&self) -> Result<SystemEnum, RgrowError> {
        Ok(match self.model.unwrap_or(MODEL_DEFAULT) {
            Model::KTAM => SystemEnum::KTAM(KTAM::try_from(self)?),
            Model::ATAM => SystemEnum::ATAM(ATAM::try_from(self)?),
            Model::OldKTAM => SystemEnum::OldKTAM(OldKTAM::try_from(self)?),
            Model::SDC => panic!("SDC not yet implemented from dynsystem create"),
        })
    }

    /// Creates an empty state, without any setup by a System.
    ///
    /// Returns
    /// -------
    /// State
    ///     An empty state.
    pub fn create_state_empty(&self) -> Result<StateEnum, RgrowError> {
        let shape = match self.size.unwrap_or(SIZE_DEFAULT) {
            Size::Single(i) => (i, i),
            Size::Pair(i) => i,
        };

        let kind = self.canvas_type.unwrap_or(CANVAS_TYPE_DEFAULT);
        let tracking = self.tracking.unwrap_or(TrackingType::None);

        Ok(StateEnum::empty(shape, kind, tracking, 1)?) // FIXME
    }

    /// Creates an empty state, without any setup by a System.
    pub fn create_state_from_canvas(&self, canvas: Array2<u32>) -> Result<StateEnum, RgrowError> {
        let kind = self.canvas_type.unwrap_or(CANVAS_TYPE_DEFAULT);
        let tracking = self.tracking.unwrap_or(TrackingType::None);

        let mut st = StateEnum::from_array(canvas.view(), kind, tracking, 1)?;

        let sys = self.create_system()?;

        sys.update_state(&mut st, &NeededUpdate::All);

        Ok(st)
    }

    /// Create a state, and set it up with a provided DynSystem.
    pub fn create_state_with_system(&self, sys: &impl DynSystem) -> Result<StateEnum, RgrowError> {
        let mut state = self.create_state_empty()?;
        sys.setup_state(&mut state)?;
        Ok(state)
    }

    /// Create a system and state
    pub fn create_system_and_state(&self) -> Result<(SystemEnum, StateEnum), RgrowError> {
        let sys = self.create_system()?;
        let state = self.create_state_with_system(&sys)?;
        Ok((sys, state))
    }

    pub fn run_window(&self) -> Result<StateEnum, RgrowError> {
        let (mut sys, mut state) = self.create_system_and_state()?;
        sys.evolve_in_window(
            &mut state,
            self.block,
            self.start_paused,
            self.get_bounds(),
            None,
            None,
        )?;
        Ok(state)
    }

    pub fn get_bounds(&self) -> EvolveBounds {
        EvolveBounds {
            size_max: self.smax,
            ..Default::default()
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

    pub(crate) hdoubletiles: Vec<(base::Tile, base::Tile)>,
    pub(crate) vdoubletiles: Vec<(base::Tile, base::Tile)>,

    pub(crate) seed: Vec<(base::CanvasLength, base::CanvasLength, base::Tile)>,

    pub(crate) glue_links: Vec<(Glue, Glue, f64)>,

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

        let ntiles = tileset.tiles.len();
        let mut tile_names: Vec<String> = Vec::with_capacity(ntiles + 1);
        let mut tile_colors = Vec::with_capacity(ntiles + 1);
        let mut tile_stoics = Vec::with_capacity(ntiles + 1);
        let mut tile_edges = Vec::with_capacity((ntiles + 1) * 4);

        let mut double_tile_edges = Vec::new();
        let mut double_tile_colors = Vec::new();
        let mut double_tile_names = Vec::new();
        let mut double_tile_stoics = Vec::new();

        let mut hdoubles = Vec::new();
        let mut vdoubles = Vec::new();

        // Push the zero state
        tile_names.push("empty".to_string());
        tile_colors.push([0, 0, 0, 0]);
        tile_stoics.push(0.);
        tile_edges.append(&mut vec![0, 0, 0, 0]);

        let mut tile_i: base::Tile = 1;
        let mut double_tile_i_offset: base::Tile = 0;

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

            let tile_color = get_color_or_random(tile.color.as_deref())?;

            let mut v: Vec<usize> = tile
                .edges
                .iter()
                .map(|te| match te {
                    GlueIdent::Name(n) => *glue_map.get_by_left(n).unwrap(),
                    GlueIdent::Num(i) => *i,
                })
                .collect();

            match &tile.shape.as_ref().unwrap_or(&TileShape::Single) {
                TileShape::Single => {
                    if v.len() != 4 {
                        return Err(ParserError::WrongNumberOfEdges {
                            name: tile_name,
                            shape: TileShape::Single,
                            num: v.len(),
                        });
                    }
                    tile_edges.append(&mut v);
                }
                TileShape::Horizontal => {
                    if v.len() != 6 {
                        return Err(ParserError::WrongNumberOfEdges {
                            name: tile_name,
                            shape: TileShape::Horizontal,
                            num: v.len(),
                        });
                    }
                    tile_edges.push(v[0]);
                    tile_edges.push(0);
                    tile_edges.push(v[4]);
                    tile_edges.push(v[5]);

                    double_tile_edges.push(v[1]);
                    double_tile_edges.push(v[2]);
                    double_tile_edges.push(v[3]);
                    double_tile_edges.push(0);

                    hdoubles.push((tile_i, double_tile_i_offset));
                    double_tile_i_offset += 1;

                    double_tile_colors.push(tile_color);
                    double_tile_stoics.push(tile_stoic);
                    double_tile_names.push(tile_name.clone());
                }
                TileShape::Vertical => {
                    if v.len() != 6 {
                        return Err(ParserError::WrongNumberOfEdges {
                            name: tile_name,
                            shape: TileShape::Vertical,
                            num: v.len(),
                        });
                    }
                    tile_edges.push(v[0]);
                    tile_edges.push(v[1]);
                    tile_edges.push(0);
                    tile_edges.push(v[5]);

                    double_tile_edges.push(0);
                    double_tile_edges.push(v[2]);
                    double_tile_edges.push(v[3]);
                    double_tile_edges.push(v[4]);

                    vdoubles.push((tile_i, double_tile_i_offset));
                    double_tile_i_offset += 1;

                    double_tile_colors.push(tile_color);
                    double_tile_stoics.push(tile_stoic);
                    double_tile_names.push(tile_name.clone());
                }
            }

            tile_names.push(tile_name);
            tile_colors.push(tile_color);
            tile_stoics.push(tile_stoic);

            tile_i += 1;
        }

        let has_duples = !hdoubles.is_empty() || !vdoubles.is_empty();

        tile_edges.append(&mut double_tile_edges);
        tile_colors.append(&mut double_tile_colors);
        tile_names.append(&mut double_tile_names);
        tile_stoics.append(&mut double_tile_stoics);

        hdoubles.iter_mut().for_each(|(_, j)| *j += tile_i);
        vdoubles.iter_mut().for_each(|(_, j)| *j += tile_i);

        let mut s = Self {
            tile_edges: Array2::from_shape_vec(
                ((tile_i + double_tile_i_offset) as usize, 4),
                tile_edges,
            )
            .unwrap(),
            tile_stoics: Array1::from_vec(tile_stoics),
            tile_names,
            tile_colors,
            glue_names: Vec::new(), // FIXME
            glue_strengths,
            has_duples,
            glue_map,
            glue_links: Vec::new(),
            hdoubletiles: hdoubles,
            vdoubletiles: vdoubles,
            seed: Vec::new(),
        };

        s.glue_links = tileset
            .glues
            .iter()
            .map(|(g1, g2, st)| (s.gpmap(g1), s.gpmap(g2), *st))
            .collect::<Vec<_>>();

        s.seed = match &tileset.seed {
            None => Vec::new(),
            Some(Seed::Single(x, y, t)) => vec![(*x, *y, s.tpmap(t))],
            Some(Seed::Multi(v)) => v
                .iter()
                .map(|(x, y, t)| {
                    (
                        *x as base::CanvasLength,
                        *y as base::CanvasLength,
                        s.tpmap(t),
                    )
                })
                .collect(),
        };

        let hdoubles = {
            match &tileset.hdoubletiles {
                Some(x) => x
                    .iter()
                    .map(|(a, b)| (s.tpmap(a), s.tpmap(b)))
                    .collect::<Vec<_>>(),
                None => Vec::new(),
            }
        };

        let vdoubles = {
            match &tileset.vdoubletiles {
                Some(x) => x
                    .iter()
                    .map(|(a, b)| (s.tpmap(a), s.tpmap(b)))
                    .collect::<Vec<_>>(),
                None => Vec::new(),
            }
        };
        s.hdoubletiles.extend(hdoubles);
        s.vdoubletiles.extend(vdoubles);

        Ok(s)
    }

    pub fn tpmap(&self, tp: &TileIdent) -> base::Tile {
        match tp {
            TileIdent::Name(x) => {
                // FIXME: fail gracefully
                self.tile_names.iter().position(|y| *y == *x).unwrap() as base::Tile
            }
            TileIdent::Num(x) => (*x).try_into().unwrap(),
        }
    }

    pub fn gpmap(&self, gp: &GlueIdent) -> base::Glue {
        match gp {
            GlueIdent::Name(x) => *self.glue_map.get_by_left(x).unwrap() as base::Glue,
            GlueIdent::Num(x) => *x,
        }
    }
}
