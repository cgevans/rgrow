use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::array;
use rgrow::{
    canvas::{CanvasPeriodic, PointSafe2},
    models::ktam::KTAM,
    state::{NullStateTracker, QuadTreeState, StateEnum},
    system::{DynSystem, System},
};
use std::hint::black_box;
use std::time::Duration;

/// Benchmark calc_committer with different configurations
fn bench_calc_committer(c: &mut Criterion) {
    let mut group = c.benchmark_group("calc_committer");

    // Set measurement time to be shorter since each calc_committer call is expensive
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // Test different grid sizes
    for grid_size in [3, 5] {
        group.bench_with_input(
            BenchmarkId::new("grid_size", grid_size),
            &grid_size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        let sys = setup_ktam_system();
                        let state = create_state_with_tiles(&sys, size);
                        let se = StateEnum::PeriodicCanvasNoTracker(state);
                        (sys, se)
                    },
                    |(mut sys, se)| black_box(sys.calc_committer(&se, 50, None, None, 5).unwrap()),
                )
            },
        );
    }

    // Test different trial counts
    for num_trials in [1, 3, 5] {
        group.bench_with_input(
            BenchmarkId::new("trials", num_trials),
            &num_trials,
            |b, &trials| {
                b.iter_with_setup(
                    || {
                        let sys = setup_ktam_system();
                        let state = create_state_with_tiles(&sys, 3);
                        let se = StateEnum::PeriodicCanvasNoTracker(state);
                        (sys, se)
                    },
                    |(mut sys, se)| {
                        black_box(sys.calc_committer(&se, 50, None, None, trials).unwrap())
                    },
                )
            },
        );
    }

    // Test different cutoff sizes
    for cutoff_size in [25, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("cutoff", cutoff_size),
            &cutoff_size,
            |b, &cutoff| {
                b.iter_with_setup(
                    || {
                        let sys = setup_ktam_system();
                        let state = create_state_with_tiles(&sys, 3);
                        let se = StateEnum::PeriodicCanvasNoTracker(state);
                        (sys, se)
                    },
                    |(mut sys, se)| {
                        black_box(sys.calc_committer(&se, cutoff, None, None, 3).unwrap())
                    },
                )
            },
        );
    }

    group.finish();
}

/// Benchmark the setup overhead separately
fn bench_setup_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup_overhead");

    group.bench_function("ktam_creation", |b| {
        b.iter(|| black_box(setup_ktam_system()))
    });

    group.bench_function("state_creation", |b| {
        b.iter_with_setup(
            || setup_ktam_system(),
            |sys| black_box(create_state_with_tiles(&sys, 3)),
        )
    });

    group.finish();
}

/// Create a KTAM system matching the test configuration
fn setup_ktam_system() -> KTAM {
    let mut sys = KTAM::new_sized(1, 1);
    sys.tile_edges = array![[0, 0, 0, 0], [1, 1, 1, 1]];
    sys.tile_concs = array![0., 1e-7];
    sys.alpha = -7.1;
    sys.glue_strengths = array![0.0, 1.0];
    sys.g_se = 4.8;
    sys.update_system();
    sys
}

/// Create a state with a k x k grid of tiles
fn create_state_with_tiles(
    sys: &KTAM,
    k: usize,
) -> QuadTreeState<CanvasPeriodic, NullStateTracker> {
    let mut state = sys
        .new_state::<QuadTreeState<CanvasPeriodic, NullStateTracker>>((32, 32))
        .unwrap();

    for r in 0..k {
        for c in 0..k {
            sys.set_safe_point(&mut state, PointSafe2((r, c)), 1);
        }
    }
    sys.setup_state(&mut state).unwrap();
    state
}

criterion_group!(benches, bench_calc_committer, bench_setup_overhead);
criterion_main!(benches);
