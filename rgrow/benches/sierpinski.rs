use criterion::{criterion_group, criterion_main, Criterion};
use rand::SeedableRng;
use rgrow::{
    canvas::CanvasPeriodic,
    models::ktam::KTAM,
    state::{NullStateTracker, QuadTreeState},
    system::{EvolveBounds, System},
    tileset::{Seed, TileSet},
    units::Second,
};

const BOUNDS10K: EvolveBounds = EvolveBounds {
    for_events: Some(10000),
    total_events: None,
    for_time: None,
    total_time: None,
    size_min: None,
    size_max: None,
    for_wall_time: None,
};

fn raw_sim_run(c: &mut Criterion) {
    let mut ts = TileSet::from_file("examples/sierpinski.yaml").unwrap();

    ts.seed = Some(Seed::Single(2045, 2045, 1.into()));

    let sys = KTAM::try_from(&ts).unwrap();

    let mut st = sys
        .new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((2048, 2048))
        .unwrap();

    let _rng = rand::rngs::SmallRng::from_os_rng();

    c.bench_function("evolve 10000 sys", |b| {
        b.iter(|| sys.evolve(&mut st, BOUNDS10K))
    });

    c.bench_function("evolve unistep", |b| {
        b.iter(|| sys.take_single_step(&mut st, Second::new(1000000.)))
    });
}
//
// fn sim_run(c: &mut Criterion) {
// let mut ts = TileSet::from_file("examples/sierpinski.yaml").unwrap();
//
// ts.seed = Some(Seed::Single(2045, 2045, 1.into()));
// ts.size = Some(rgrow::tileset::Size::Single(2048));
// ts.model = Some(rgrow::tileset::Model::KTAM);
//
// let mut sim = TileSet::into_simulation(&ts).unwrap();
// sim.add_state().unwrap();
//
// c.bench_function("evolve 10000 sim", |b| b.iter(|| sim.evolve(0, BOUNDS10K)));
//
// ts.model = Some(rgrow::tileset::Model::OldKTAM);
// let mut sim = TileSet::into_simulation(&ts).unwrap();
//
// sim.add_state().unwrap();
//
// c.bench_function("evolve 10000 old", |b| b.iter(|| sim.evolve(0, BOUNDS10K)));
// }
//
criterion_group!(benches, raw_sim_run);
criterion_main!(benches);
