//! Profiling driver: runs each model under representative parameters in a loop.
//!
//! Run under samply with:
//!     samply record -- target/profiling/examples/perf_drive [model]
//!
//! Models: ktam | ktam_periodic | atam | sdc | kblock | all (default)
use rgrow::{
    canvas::{CanvasPeriodic, CanvasSquare},
    models::ktam::KTAM,
    state::{NullStateTracker, QuadTreeState, StateStatus},
    system::{EvolveBounds, System},
    tileset::{Seed, TileSet},
};
use std::env;
use std::time::Instant;

const N_EVENTS: u64 = 20_000_000;

fn run_ktam_sierpinski_periodic() {
    let mut ts = TileSet::from_file("../examples/sierpinski.yaml").unwrap();
    ts.seed = Some(Seed::Single(2045, 2045, 1.into()));
    let sys = KTAM::try_from(&ts).unwrap();
    let mut st = sys
        .new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((2048, 2048))
        .unwrap();

    let bounds = EvolveBounds {
        for_events: Some(N_EVENTS),
        ..Default::default()
    };

    let t0 = Instant::now();
    sys.evolve(&mut st, bounds).unwrap();
    let dt = t0.elapsed();
    let n_tiles = st.n_tiles();
    println!(
        "KTAM Sierpinski Periodic 2048x2048: {} events in {:.2?} = {:.1} ns/event, n_tiles={}",
        N_EVENTS,
        dt,
        dt.as_nanos() as f64 / N_EVENTS as f64,
        n_tiles
    );
}

fn run_ktam_sierpinski_square() {
    let mut ts = TileSet::from_file("../examples/sierpinski.yaml").unwrap();
    ts.seed = Some(Seed::Single(509, 509, 1.into()));
    let sys = KTAM::try_from(&ts).unwrap();
    let mut st = sys
        .new_state::<QuadTreeState<CanvasSquare, NullStateTracker>>((512, 512))
        .unwrap();

    let bounds = EvolveBounds {
        for_events: Some(N_EVENTS),
        ..Default::default()
    };

    let t0 = Instant::now();
    sys.evolve(&mut st, bounds).unwrap();
    let dt = t0.elapsed();
    let n_tiles = st.n_tiles();
    println!(
        "KTAM Sierpinski Square 512x512: {} events in {:.2?} = {:.1} ns/event, n_tiles={}",
        N_EVENTS,
        dt,
        dt.as_nanos() as f64 / N_EVENTS as f64,
        n_tiles
    );
}

fn run_atam_sierpinski() {
    use rgrow::tileset::Model;
    let mut ts = TileSet::from_file("../examples/sierpinski.yaml").unwrap();
    // Larger canvas so ATAM doesn't saturate within the event budget.
    ts.seed = Some(Seed::Single(2045, 2045, 1.into()));
    ts.model = Some(Model::ATAM);
    let sys = rgrow::models::atam::ATAM::try_from(&ts).unwrap();
    let mut st = sys
        .new_state::<QuadTreeState<CanvasSquare, NullStateTracker>>((2048, 2048))
        .unwrap();

    let bounds = EvolveBounds {
        for_events: Some(N_EVENTS),
        ..Default::default()
    };
    let t0 = Instant::now();
    sys.evolve(&mut st, bounds).unwrap();
    let dt = t0.elapsed();
    let n_tiles = st.n_tiles();
    println!(
        "ATAM Sierpinski Square 2048x2048: {} events in {:.2?} = {:.1} ns/event, n_tiles={}",
        N_EVENTS,
        dt,
        dt.as_nanos() as f64 / N_EVENTS as f64,
        n_tiles
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let target = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    match target {
        "ktam" => run_ktam_sierpinski_square(),
        "ktam_periodic" => run_ktam_sierpinski_periodic(),
        "atam" => run_atam_sierpinski(),
        "all" => {
            run_ktam_sierpinski_periodic();
            run_ktam_sierpinski_square();
            run_atam_sierpinski();
        }
        other => panic!("Unknown target: {other}"),
    }
}
