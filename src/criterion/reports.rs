use std::{hint::black_box, time::Duration};

use criterion::{Criterion, Throughput};

use pickems::{
    datatypes::{Iterations, Set},
    reporting::{AssessReport, BasicReport, NullReport, PicksReport, StrengthReport},
    simulation::Simulation,
};

const ITERATIONS: u64 = 10_000;

pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("reports");
    group.significance_level(0.1);
    group.noise_threshold(0.02);
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(ITERATIONS));

    group.bench_function("null", |b| {
        let sim = Simulation::dummy(Iterations::new(ITERATIONS));
        b.iter(|| black_box(black_box(&sim).bench_test(NullReport)));
    });

    group.bench_function("basic", |b| {
        let sim = Simulation::dummy(Iterations::new(ITERATIONS));
        b.iter(|| black_box(black_box(&sim).bench_test(BasicReport::default())));
    });

    group.bench_function("picks", |b| {
        let sim = Simulation::dummy(Iterations::new(ITERATIONS));
        b.iter(|| black_box(black_box(&sim).bench_test(PicksReport::default())));
    });

    group.bench_function("strength", |b| {
        let sim = Simulation::dummy(Iterations::new(ITERATIONS));
        b.iter(|| black_box(black_box(&sim).bench_test(StrengthReport::default())));
    });

    group.bench_function("assess", |b| {
        let sim = Simulation::dummy(Iterations::new(ITERATIONS));
        let tz = Set::from(1);
        let adv = Set::from(2);
        let zt = Set::from(3);
        b.iter(|| black_box(black_box(&sim).bench_test(AssessReport::new(tz, adv, zt))));
    });

    group.finish();
}
