use criterion::{Criterion, criterion_group, criterion_main};
use pickems::Simulation;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("group");
    group.sample_size(40);
    group.bench_function("Simulation", |b| b.iter(|| Simulation::bench_test(10000)));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
