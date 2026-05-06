//! P16 investigation: measure evolve_states scaling with thread count.
//!
//! Build:
//!     cargo build --release -p rgrow --example evolve_states_scaling
//!
//! Use RAYON_NUM_THREADS to control rayon parallelism. The binary builds K
//! identical KTAM-sierpinski states, runs a warmup, then times
//! `sys.evolve_states(...)` for E events per state.
//!
//! Output is one line of TSV-ish data per run for easy aggregation:
//!     threads=N k=K events=E wall_ms=W ns_per_event=Ne ns_per_event_per_state=Nes total_tiles=T
//!
//! Args: --k <K> --events <E> [--canvas square|periodic] [--size <N>]
use rgrow::{
    canvas::{CanvasPeriodic, CanvasSquare},
    models::ktam::KTAM,
    state::{NullStateTracker, QuadTreeState, StateEnum, StateStatus},
    system::{EvolveBounds, System},
    tileset::{Seed, TileSet},
};
use std::env;
use std::time::Instant;

fn parse_args() -> (usize, u64, &'static str, usize) {
    let args: Vec<String> = env::args().collect();
    let mut k: usize = 8;
    let mut events: u64 = 200_000;
    let mut canvas: &'static str = "square";
    let mut size: usize = 512;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--k" => {
                k = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--events" => {
                events = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--canvas" => {
                canvas = if args[i + 1] == "periodic" {
                    "periodic"
                } else {
                    "square"
                };
                i += 2;
            }
            "--size" => {
                size = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => i += 1,
        }
    }
    (k, events, canvas, size)
}

fn build_sys(_canvas: &str, size: usize) -> KTAM {
    let mut ts = TileSet::from_file("../examples/sierpinski.yaml").unwrap();
    let center = (size - 3) as u64;
    ts.seed = Some(Seed::Single(
        center.try_into().unwrap(),
        center.try_into().unwrap(),
        1.into(),
    ));
    KTAM::try_from(&ts).unwrap()
}

fn main() {
    let (k, events, canvas, size) = parse_args();
    let threads = rayon::current_num_threads();

    let sys = build_sys(canvas, size);

    // Build K identical states. Use StateEnum so we exercise the same dispatch
    // path the Python `evolve_states` users would hit through FFS / committor.
    let mut states: Vec<StateEnum> = (0..k)
        .map(|_| match canvas {
            "periodic" => StateEnum::PeriodicCanvasNoTracker(
                sys.new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((size, size))
                    .unwrap(),
            ),
            _ => StateEnum::SquareCanvasNullTracker(
                sys.new_state::<QuadTreeState<CanvasSquare, NullStateTracker>>((size, size))
                    .unwrap(),
            ),
        })
        .collect();

    // Warmup: evolve each state for 1000 events sequentially so cache & friends
    // tables are populated before the timed parallel section.
    let warmup_bounds = EvolveBounds {
        for_events: Some(1000),
        ..Default::default()
    };
    for s in &mut states {
        sys.evolve(s, warmup_bounds).unwrap();
    }

    let bounds = EvolveBounds {
        for_events: Some(events),
        ..Default::default()
    };

    let t0 = Instant::now();
    let outcomes = sys.evolve_states(&mut states, bounds);
    let dt = t0.elapsed();

    for o in &outcomes {
        if let Err(e) = o {
            eprintln!("warning: evolve outcome error: {e:?}");
        }
    }

    let total_tiles: u64 = states.iter().map(|s| s.n_tiles() as u64).sum();
    let total_events = events * (k as u64);
    let wall_ms = dt.as_secs_f64() * 1000.0;
    let ns_per_event = (dt.as_nanos() as f64) / (total_events as f64);
    let ns_per_event_per_state = (dt.as_nanos() as f64) / (events as f64);

    println!(
        "threads={} k={} events={} canvas={} size={} wall_ms={:.2} ns_per_event={:.1} ns_per_event_per_state={:.1} total_tiles={}",
        threads,
        k,
        events,
        canvas,
        size,
        wall_ms,
        ns_per_event,
        ns_per_event_per_state,
        total_tiles
    );
}
