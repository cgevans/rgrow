extern crate rgrow;

use anyhow::{Context, Result};

use std::fs::File;

use rgrow::{
    canvas::{CanvasPeriodic, PointSafe2},
    state::{NullStateTracker, QuadTreeState, StateStatus},
    system::{EvolveBounds, FissionHandling, System},
    tileset::{FromTileSet, ParsedSeed, Size, TileSet},
};

fn test_sim(ts: &TileSet) -> Result<()> {
    let mut sim = ts.into_simulation()?;
    sim.add_state()?;
    sim.evolve(
        0,
        EvolveBounds {
            for_events: Some(10),
            ..Default::default()
        },
    )?;
    Ok(())
}

#[test]
fn parser_test() -> Result<()> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir("./examples/")? {
        let entry = entry?;
        let path = entry.path();
        if let Some("yaml") = path.extension().map(|s| s.to_str().unwrap()) {
            paths.push(path);
        }
    }
    for p in paths.drain(..) {
        println!("path: {p:?}");
        let ts = TileSet::from_file(&p)?;
        test_sim(&ts).context(format!("Failed to simulate: {}", p.to_string_lossy()))?;
    }

    paths.clear();
    for entry in std::fs::read_dir("./examples/xgrow-format/")? {
        let entry = entry?;
        let path = entry.path();
        println!("path: {path:?}");
        if let Some("tiles") = path.extension().map(|s| s.to_str().unwrap()) {
            paths.push(path);
        }
    }
    for p in paths.drain(..) {
        let ts = TileSet::from_file(&p)?;
        test_sim(&ts).context(format!("Failed to simulate: {}", p.to_string_lossy()))?;
    }
    Ok(())
}

#[test]
fn atam_test() -> Result<()> {
    let mut ts = get_sierpinski()?;

    ts.options.model = rgrow::tileset::Model::ATAM;
    ts.options.size = Size::Single(64);
    let p = PointSafe2((60, 60));
    ts.options.seed = ParsedSeed::Single(p.0 .0, p.0 .1, 1.into());

    let mut at = ts.into_simulation()?;

    at.add_state()?;

    at.evolve(
        0,
        EvolveBounds {
            size_max: Some(500),
            for_events: Some(2000),
            ..Default::default()
        },
    )?;

    let state = at.state_ref(0);

    let sr = state; //.read().unwrap();

    assert!(sr.ntiles() == 500);

    assert!(sr.tile_at_point(PointSafe2((p.0 .0 - 7, p.0 .1 - 7))) == 4);
    assert!(sr.tile_at_point(PointSafe2((p.0 .0 - 8, p.0 .1 - 8))) == 5);

    Ok(())
}

#[test]
fn ktam_test() -> Result<()> {
    let mut ts = get_sierpinski()?;

    ts.options.model = rgrow::tileset::Model::KTAM;
    ts.options.size = Size::Single(64);
    ts.options.seed = ParsedSeed::Single(60, 60, 1.into());

    let mut sys = rgrow::models::ktam::KTAM::from_tileset(&ts)?;

    let mut st = sys.new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((64, 64))?;

    sys.g_se = 8.1;
    sys.update_system();

    sys.evolve(
        &mut st,
        EvolveBounds {
            for_events: Some(20000),
            size_max: Some(210),
            ..Default::default()
        },
    )?;

    assert!(st.ntiles() > 200);

    sys.g_se = 7.8;
    sys.update_system();

    sys.evolve(
        &mut st,
        EvolveBounds {
            for_events: Some(100000),
            size_min: Some(10),
            ..Default::default()
        },
    )?;

    assert!(st.ntiles() < 100);

    // let state = at.state_ref(0);

    Ok(())
}

#[test]
fn ktam_barish_test() -> Result<()> {
    let mut ts = TileSet::from_file("examples/barish-perfect.yaml")?;

    ts.options.model = rgrow::tileset::Model::KTAM;
    ts.options.gse = 8.5;
    ts.options.gmc = 16.0;

    let mut sim = ts.into_simulation()?;

    let si = sim.add_state()?;

    sim.evolve(
        si,
        EvolveBounds {
            for_events: Some(20000),
            size_max: Some(220),
            ..Default::default()
        },
    )?;

    assert!(sim.state_ref(si).ntiles() > 200);

    Ok(())
}

#[test]
fn oldktam_test() -> Result<()> {
    let mut ts = get_sierpinski()?;

    ts.options.model = rgrow::tileset::Model::OldKTAM;
    ts.options.size = Size::Single(64);
    ts.options.seed = ParsedSeed::Single(60, 60, 1.into());
    ts.options.gse = 8.1;

    let sys = rgrow::models::oldktam::OldKTAM::from_tileset(&ts)?;

    let mut st = sys.new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((64, 64))?;

    sys.evolve(
        &mut st,
        EvolveBounds {
            for_events: Some(20000),
            size_max: Some(210),
            ..Default::default()
        },
    )?;

    assert!(st.ntiles() > 200);

    ts.options.gse = 7.8;
    let sys = rgrow::models::oldktam::OldKTAM::from_tileset(&ts)?;

    sys.evolve(
        &mut st,
        EvolveBounds {
            for_events: Some(100000),
            size_min: Some(10),
            ..Default::default()
        },
    )?;

    assert!(st.ntiles() < 100);

    // let state = at.state_ref(0);

    Ok(())
}

#[test]
fn simple_fission_test() -> Result<()> {
    let mut ts = TileSet::from_file("examples/fission-small-ribbon.yaml")?;

    ts.options.fission = FissionHandling::NoFission;
    let mut sim = ts.into_simulation()?;
    sim.add_state()?;
    sim.evolve(
        0,
        EvolveBounds {
            for_time: Some(1000.),
            ..Default::default()
        },
    )?;
    let state = sim.state_ref(0);
    assert!(state.ntiles() > 800); //.read().unwrap()

    ts.options.fission = FissionHandling::KeepSeeded;
    let mut sim = ts.into_simulation()?;
    sim.add_state()?;
    sim.evolve(
        0,
        EvolveBounds {
            for_time: Some(1000.),
            ..Default::default()
        },
    )?;
    let state = sim.state_ref(0);
    assert!(state.ntiles() < 500); // .read().unwrap()
    Ok(())
}

#[test]
fn nucrate_test() -> Result<()> {
    let mut ts: TileSet = serde_yaml::from_reader(File::open("examples/barish-perfect.yaml")?)?;

    ts.options.alpha = -7.1;
    ts.options.gse = 5.7;
    ts.options.gmc = 9.7;
    ts.options.model = rgrow::tileset::Model::KTAM;
    ts.options.canvas_type = rgrow::tileset::CanvasType::Periodic;
    ts.options.fission = rgrow::system::FissionHandling::KeepLargest;

    let conf = rgrow::ffs::FFSRunConfig {
        max_configs: 100,
        ..Default::default()
    };

    let _result = ts.run_ffs(&conf)?;

    Ok(())
}

fn get_sierpinski() -> Result<TileSet> {
    serde_yaml::from_reader(File::open("examples/sierpinski.yaml")?)
        .context("Failure opening sierpinski example.")
}
