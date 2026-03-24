use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rgrow::{
    canvas::{Canvas, CanvasPeriodic, CanvasSquare},
    models::ktam::KTAM,
    state::{NullStateTracker, QuadTreeState, StateWithCreate},
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
    let mut ts = TileSet::from_file("../examples/sierpinski.yaml").unwrap();

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
fn state_copy_benchmarks(c: &mut Criterion) {
    use ndarray::array;

    let mut group = c.benchmark_group("state_copy");

    // Benchmark sparse copy at different canvas sizes with ~5 tiles
    for size in [16usize, 32, 128] {
        let mut system = KTAM::new_sized(5, 2);
        system.tile_edges = array![
            [0, 0, 0, 0], // tile 0 (empty)
            [1, 0, 0, 0], // tile 1
            [0, 0, 1, 0], // tile 2
            [1, 1, 0, 0], // tile 3
            [0, 0, 1, 1], // tile 4
        ];
        system.glue_strengths = array![0., 1.0];
        system.tile_concs = array![0., 1e-7, 1e-7, 1e-7, 1e-7];
        system.g_se = 8.0;
        system.alpha = 7.1;
        system.update_system();

        let canvas_size = size + 4; // CanvasSquare adds 4 for border
        let mut state: QuadTreeState<CanvasSquare, NullStateTracker> =
            system.new_state((canvas_size, canvas_size)).unwrap();
        let center = state.center();

        // Place ~5 tiles in a cluster
        let pe = rgrow::canvas::PointSafe2(state.move_sa_e(center).0);
        let ps = rgrow::canvas::PointSafe2(state.move_sa_s(center).0);
        let pn = rgrow::canvas::PointSafe2(state.move_sa_n(center).0);
        let pw = rgrow::canvas::PointSafe2(state.move_sa_w(center).0);
        system.set_safe_point(&mut state, center, 1);
        system.set_safe_point(&mut state, pe, 2);
        system.set_safe_point(&mut state, ps, 3);
        system.set_safe_point(&mut state, pn, 4);
        system.set_safe_point(&mut state, pw, 1);

        group.bench_with_input(
            BenchmarkId::new("sparse", format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter_batched_ref(
                    || QuadTreeState::empty((canvas_size, canvas_size)).unwrap(),
                    |target| system.clone_state_into_empty_state(&state, target),
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("clone_from", format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter_batched_ref(
                    || QuadTreeState::empty((canvas_size, canvas_size)).unwrap(),
                    |target| target.clone_from(&state),
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Hybrid: sparse rate tree copy + naive canvas memcpy.
        // This avoids traversing the canvas through the quadtree at the cost
        // of copying the full canvas array (much smaller than the rate tree).
        group.bench_with_input(
            BenchmarkId::new("sparse_rates_naive_canvas", format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter_batched_ref(
                    || QuadTreeState::empty((canvas_size, canvas_size)).unwrap(),
                    |target| {
                        // Naive canvas copy (ndarray assign = optimized memcpy)
                        target.raw_array_mut().assign(&state.raw_array());
                        // Sparse rate tree copy (also writes canvas tiles
                        // redundantly, but those cache lines are already hot)
                        system.clone_state_into_empty_state(&state, target);
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    // Benchmark with duples on 32x32
    {
        let mut system = KTAM::new_sized(4, 2);
        system.tile_edges = array![
            [0, 0, 0, 0], // tile 0 (empty)
            [1, 0, 0, 0], // tile 1 (single)
            [1, 0, 0, 0], // tile 2 (real duple half)
            [0, 0, 1, 0], // tile 3 (fake duple half)
        ];
        system.glue_strengths = array![0., 1.0];
        system.tile_concs = array![0., 1e-7, 1e-7, 1e-7];
        system.g_se = 8.0;
        system.alpha = 7.1;
        system.set_duples(vec![(2, 3)], vec![]);

        let mut state: QuadTreeState<CanvasSquare, NullStateTracker> =
            system.new_state((36, 36)).unwrap();
        let center = state.center();

        system.set_safe_point(&mut state, center, 1);
        // Place duple at odd column to test cross-quadrant
        let dp = rgrow::canvas::PointSafe2((center.0 .0, center.0 .1 + 1));
        system.set_safe_point(&mut state, dp, 2);

        group.bench_function("sparse_with_duples_32x32", |b| {
            b.iter_batched_ref(
                || QuadTreeState::empty((36, 36)).unwrap(),
                |target| system.clone_state_into_empty_state(&state, target),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, raw_sim_run, state_copy_benchmarks);
criterion_main!(benches);
