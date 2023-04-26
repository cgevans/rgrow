use criterion::Criterion;
use rgrow::tileset::{ParsedSeed, TileSet};

fn bench_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("window");
    group.significance_level(0.1).sample_size(10);

    let mut ts = TileSet::from_file("examples/sierpinski.yaml").unwrap();

    ts.options.seed = ParsedSeed::Single(1020, 1020, 1.into());
    ts.options.size = rgrow::tileset::Size::Single(1024);
    ts.options.model = rgrow::tileset::Model::KTAM;
    ts.options.smax = Some(20000);

    group.bench_function("window", |b| b.iter(|| rgrow::ui::run_window(&ts)));
}

criterion::criterion_group!(benches, bench_window);
criterion::criterion_main!(benches);
