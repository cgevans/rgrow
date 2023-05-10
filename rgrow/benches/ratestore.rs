use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use rand::{seq::SliceRandom, Rng};
use rgrow::{
    canvas::PointSafeHere,
    ratestore::{CreateSizedRateStore, QuadTreeSquareArray, RateStore},
};

fn ratestore_qsta_update(c: &mut Criterion) {
    // Create a new RateStore
    let mut rs = QuadTreeSquareArray::new_with_size(256, 256);
    let mut rs_large = rs.clone();
    let mut rs_single = rs.clone();

    let rng = rand::thread_rng();
    let it = rng.sample_iter(rand::distributions::Uniform::new(0.0, 1.0));

    let allchanges = (0..256usize)
        .flat_map(|x| (0..256usize).map(move |y| (x, y)))
        .zip(it)
        .map(|((x, y), r)| (PointSafeHere((x, y)), r))
        .collect::<Vec<_>>();

    let allchanges_shuffled = {
        let mut v = allchanges.clone();
        v.shuffle(&mut rand::thread_rng());
        v
    };

    println!("small update");

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("ratestore-update-all");
    group.plot_config(plot_config.clone());
    for (pn, pv) in &[
        ("all", &allchanges[..]),
        ("all_shuffle", &allchanges_shuffled[..]),
    ] {
        group.bench_with_input(BenchmarkId::new("small update", pn), &pv, |b, a| {
            b.iter(|| rs._update_multiple_small(a))
        });

        group.bench_with_input(BenchmarkId::new("large update", pn), &pv, |b, a| {
            b.iter(|| rs_large._update_multiple_large(a))
        });

        group.bench_with_input(BenchmarkId::new("all update", pn), &pv, |b, a| {
            b.iter(|| rs_large._update_multiple_all(a))
        });

        group.bench_with_input(BenchmarkId::new("single update", pn), &pv, |b, a| {
            b.iter(|| {
                for (p, r) in a.iter() {
                    rs_single.update_point(p.0, *r);
                }
            })
        });
    }

    group.finish();

    let mut group = c.benchmark_group("ratestore-update-sized");
    group.plot_config(plot_config.clone());

    for &s in &[8, 16, 32, 64, 128, 256, 512] {
        if s < 2000 {
            group.bench_with_input(
                BenchmarkId::new("small update", s),
                &allchanges[0..s],
                |b, a| b.iter(|| rs._update_multiple_small(a)),
            );
        }
        group.bench_with_input(
            BenchmarkId::new("large update", s),
            &allchanges[0..s],
            |b, a| b.iter(|| rs_large._update_multiple_large(a)),
        );

        group.bench_with_input(
            BenchmarkId::new("all update", s),
            &allchanges[0..s],
            |b, a| b.iter(|| rs_large._update_multiple_all(a)),
        );

        group.bench_with_input(
            BenchmarkId::new("single update", s),
            &allchanges[0..s],
            |b, a| {
                b.iter(|| {
                    for (p, r) in a.iter() {
                        rs_single.update_point(p.0, *r);
                    }
                })
            },
        );
    }

    group.finish();

    let mut group = c.benchmark_group("ratestore-update-combined");
    group.plot_config(plot_config);
    for &s in &[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        group.bench_with_input(
            BenchmarkId::new("combined-update", s),
            &allchanges[0..s],
            |b, a| b.iter(|| rs.update_multiple(a)),
        );
    }

    group.finish();
}

criterion_group!(benches, ratestore_qsta_update);
criterion_main!(benches);
