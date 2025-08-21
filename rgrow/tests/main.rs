extern crate rgrow;

use anyhow::{Context, Result};
use ndarray::array;

use std::fs::File;

use rgrow::{
    canvas::{Canvas, CanvasPeriodic, PointSafe2},
    models::ktam::KTAM,
    state::{NullStateTracker, QuadTreeState, StateEnum, StateStatus},
    system::{DynSystem, EvolveBounds, FissionHandling, System},
    tileset::{Seed, Size, TileSet},
};

fn test_sim(ts: &TileSet) -> Result<()> {
    let (sys, mut state) = ts.create_system_and_state()?;
    sys.evolve(
        &mut state,
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

    ts.model = Some(rgrow::tileset::Model::ATAM);
    ts.size = Some(Size::Single(64));
    let p = PointSafe2((60, 60));
    ts.seed = Some(Seed::Single(p.0 .0, p.0 .1, 1.into()));

    let (sys, mut state) = ts.create_system_and_state()?;

    sys.evolve(
        &mut state,
        EvolveBounds {
            size_max: Some(500),
            for_events: Some(2000),
            ..Default::default()
        },
    )?;

    assert!(state.n_tiles() == 500);

    assert!(state.tile_at_point(PointSafe2((p.0 .0 - 7, p.0 .1 - 7))) == 4);
    assert!(state.tile_at_point(PointSafe2((p.0 .0 - 8, p.0 .1 - 8))) == 5);

    Ok(())
}

#[test]
fn ktam_test() -> Result<()> {
    let mut ts = get_sierpinski()?;

    ts.model = Some(rgrow::tileset::Model::KTAM);
    ts.size = Some(Size::Single(64));
    ts.seed = Some(Seed::Single(60, 60, 1.into()));

    let mut sys = rgrow::models::ktam::KTAM::try_from(&ts)?;

    let mut st = sys.new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((64, 64))?;

    sys.g_se = 8.1;
    sys.update_system();

    System::evolve(
        &sys,
        &mut st,
        EvolveBounds {
            for_events: Some(20000),
            size_max: Some(210),
            ..Default::default()
        },
    )?;

    assert!(st.n_tiles() > 200);

    sys.g_se = 7.8;
    sys.update_system();

    System::evolve(
        &sys,
        &mut st,
        EvolveBounds {
            for_events: Some(100000),
            size_min: Some(10),
            ..Default::default()
        },
    )?;

    assert!(st.n_tiles() < 100);

    // let state = at.state_ref(0);

    Ok(())
}

#[test]
fn ktam_barish_test() -> Result<()> {
    let mut ts = TileSet::from_file("examples/barish-perfect.yaml")?;

    print!("ts: {ts:?}");

    ts.model = Some(rgrow::tileset::Model::KTAM);
    ts.gse = Some(8.5);
    ts.gmc = Some(16.0);

    let (sys, mut state) = ts.create_system_and_state()?;

    sys.evolve(
        &mut state,
        EvolveBounds {
            for_events: Some(20000),
            size_max: Some(220),
            ..Default::default()
        },
    )?;

    println!("ntiles: {}", state.n_tiles());
    assert!(state.n_tiles() > 200);

    Ok(())
}

#[test]
fn oldktam_test() -> Result<()> {
    let mut ts = get_sierpinski()?;

    ts.model = Some(rgrow::tileset::Model::OldKTAM);
    ts.size = Some(Size::Single(64));
    ts.seed = Some(Seed::Single(60, 60, 1.into()));
    ts.gse = Some(8.1);

    let sys = rgrow::models::oldktam::OldKTAM::try_from(&ts)?;

    let mut st = sys.new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((64, 64))?;

    System::evolve(
        &sys,
        &mut st,
        EvolveBounds {
            for_events: Some(20000),
            size_max: Some(210),
            ..Default::default()
        },
    )?;

    assert!(st.n_tiles() > 200);

    ts.gse = Some(7.8);
    let sys = rgrow::models::oldktam::OldKTAM::try_from(&ts)?;

    System::evolve(
        &sys,
        &mut st,
        EvolveBounds {
            for_events: Some(100000),
            size_min: Some(10),
            ..Default::default()
        },
    )?;

    assert!(st.n_tiles() < 100);

    // let state = at.state_ref(0);

    Ok(())
}

#[test]
fn simple_fission_test() -> Result<()> {
    let mut ts = TileSet::from_file("examples/fission-small-ribbon.yaml")?;

    ts.fission = Some(FissionHandling::NoFission);
    let (sys, mut state) = ts.create_system_and_state()?;
    sys.evolve(
        &mut state,
        EvolveBounds {
            for_time: Some(1000.),
            ..Default::default()
        },
    )?;
    assert!(state.n_tiles() > 800); //.read().unwrap()

    ts.fission = Some(FissionHandling::KeepSeeded);
    let (sys, mut state) = ts.create_system_and_state()?;
    sys.evolve(
        &mut state,
        EvolveBounds {
            for_time: Some(1000.),
            ..Default::default()
        },
    )?;
    assert!(state.n_tiles() < 500); // .read().unwrap()
    Ok(())
}

#[test]
fn nucrate_test() -> Result<()> {
    let mut ts: TileSet = serde_yaml::from_reader(File::open("examples/barish-perfect.yaml")?)?;

    ts.alpha = Some(-7.1);
    ts.gse = Some(5.7);
    ts.gmc = Some(9.7);
    ts.model = Some(rgrow::tileset::Model::KTAM);
    ts.canvas_type = Some(rgrow::tileset::CanvasType::Periodic);
    ts.fission = Some(rgrow::system::FissionHandling::KeepLargest);

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

#[test]
fn test_calc_committer() -> Result<()> {
    // In these conditions, the max energy square is k=8.
    let mut sys = KTAM::new_sized(1, 1);
    sys.tile_edges = array![[0, 0, 0, 0], [1, 1, 1, 1]];
    sys.tile_concs = array![0., 1e-7];
    sys.alpha = -7.1;
    sys.glue_strengths = array![0.0, 1.0];

    sys.g_se = 4.8;

    sys.update_system();

    let mut state = sys.new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((32, 32))?;

    let k = 14;
    for r in 0..k {
        for c in 0..k {
            sys.set_safe_point(&mut state, PointSafe2((r, c)), 1);
        }
    }
    sys.setup_state(&mut state)?;

    let se = StateEnum::PeriodicCanvasNoTracker(state);

    let committer = sys.calc_committer(&se, 200, None, None, 100)?;

    println!("committer k=14: {committer}");
    assert!(
        (committer > 0.9),
        "committer out of expected range: {committer}"
    );

    let mut state = sys.new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((32, 32))?;

    let k = 8;
    for r in 0..k {
        for c in 0..k {
            sys.set_safe_point(&mut state, PointSafe2((r, c)), 1);
        }
    }
    sys.setup_state(&mut state)?;

    let se = StateEnum::PeriodicCanvasNoTracker(state);

    let committer = sys.calc_committer(&se, 200, None, None, 100)?;

    println!("committer k=8: {committer}");
    assert!(
        (committer > 0.5) && (committer < 0.9),
        "committer out of expected range: {committer}"
    );

    let mut state = sys.new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((32, 32))?;

    let k = 5;
    for r in 0..k {
        for c in 0..k {
            sys.set_safe_point(&mut state, PointSafe2((r, c)), 1);
        }
    }
    sys.setup_state(&mut state)?;

    let se = StateEnum::PeriodicCanvasNoTracker(state);

    let committer = sys.calc_committer(&se, 200, None, None, 100)?;

    println!("committer k=5: {committer}");
    assert!(
        (committer < 0.1),
        "committer out of expected range: {committer}"
    );

    Ok(())
}
