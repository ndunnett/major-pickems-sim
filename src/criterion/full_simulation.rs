use std::{hint::black_box, time::Duration};

use criterion::{Criterion, Throughput};

use pickems::{datatypes::Iterations, reporting::NullReport, simulation::Simulation};

const PARALLEL_ITERATIONS: u64 = 1_000_000;
const SINGLE_THREAD_ITERATIONS: u64 = 20_000;

pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_simulation");
    group.significance_level(0.1);
    group.noise_threshold(0.02);

    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(SINGLE_THREAD_ITERATIONS));
    group.bench_function("single_thread", |b| {
        let sim = Simulation::dummy(Iterations::new(SINGLE_THREAD_ITERATIONS));
        b.iter(|| black_box(black_box(&sim).bench_test(NullReport)));
    });

    group.sample_size(20);
    group.measurement_time(Duration::from_mins(1));
    group.throughput(Throughput::Elements(PARALLEL_ITERATIONS));
    group.bench_function("parallel", |b| {
        let sim = Simulation::dummy(Iterations::new(PARALLEL_ITERATIONS));
        b.iter(|| black_box(black_box(&sim).run(NullReport)));
    });

    group.finish();
}
