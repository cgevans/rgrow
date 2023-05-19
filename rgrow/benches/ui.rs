use criterion::Criterion;
use rgrow::tileset::{Seed, TileSet};

fn bench_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("window");
    group.significance_level(0.1).sample_size(10);

    let mut ts = TileSet::from_file("examples/sierpinski.yaml").unwrap();

    ts.seed = Some(Seed::Single(1020, 1020, 1.into()));
    ts.size = Some(rgrow::tileset::Size::Single(1024));
    ts.model = Some(rgrow::tileset::Model::KTAM);
    ts.smax = Some(20000);

    group.bench_function("window", |b| b.iter(|| ts.run_window()));
}

criterion::criterion_group!(benches, bench_window);
criterion::criterion_main!(benches);
