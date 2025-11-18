use criterion::{Criterion, criterion_group, criterion_main};
use pickems::{NullReport, Simulation};

fn bench(c: &mut Criterion) {
    let sim = Simulation::dummy(100000);
    let mut group = c.benchmark_group("Simulation");
    group.sample_size(10);
    group.bench_function("run", |b| b.iter(|| sim.clone().run::<NullReport>()));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
