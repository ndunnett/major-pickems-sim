use criterion::{Criterion, criterion_group, criterion_main};

use pickems::{reporting::NullReport, simulation::Simulation};

fn bench(c: &mut Criterion) {
    let sim = Simulation::dummy(1_000_000);
    let mut group = c.benchmark_group("Simulation");
    group.sample_size(100);

    group.bench_function("parallel", |b| {
        b.iter(|| sim.clone().run(NullReport));
    });

    let sim = Simulation::dummy(50_000);

    group.bench_function("single_thread", |b| {
        b.iter(|| sim.clone().bench_test(NullReport));
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
