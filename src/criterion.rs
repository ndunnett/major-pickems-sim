use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use pickems::{
    datatypes::{Rating, Set},
    reporting::NullReport,
    simulation::{Simulation, probabilities, seeding},
};

fn bench_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation");

    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .bench_function("parallel", |b| {
            let sim = Simulation::dummy(1_000_000);
            b.iter(|| sim.clone().run(NullReport));
        })
        .bench_function("single_thread", |b| {
            let sim = Simulation::dummy(50_000);
            b.iter(|| sim.clone().bench_test(NullReport));
        });

    group.finish();
}

struct SeedingCase {
    name: &'static str,
    remaining: Set,
    diffs: [i8; 16],
    opponents: [Set; 16],
}

fn opponent_fixture(bits: u16) -> [Set; 16] {
    std::array::from_fn(|i| {
        let rotate = u32::try_from(i).unwrap();
        Set::from(bits.rotate_left(rotate))
    })
}

fn seeding_cases() -> [SeedingCase; 5] {
    [
        SeedingCase {
            name: "16_teams",
            remaining: Set::from(0xFFFF),
            diffs: [0; 16],
            opponents: opponent_fixture(0b0001_0010_1010_0101),
        },
        SeedingCase {
            name: "12_teams",
            remaining: Set::from(0x0FFF),
            diffs: [3, 3, 2, 2, 2, 1, 1, 1, -1, -1, -1, -2, -2, -2, -3, -3],
            opponents: opponent_fixture(0b0011_0101_1000_1110),
        },
        SeedingCase {
            name: "8_teams",
            remaining: Set::from(0x5555),
            diffs: [1, 1, 1, 1, -1, -1, -1, -1, 2, 2, -2, -2, 3, -3, 0, 0],
            opponents: opponent_fixture(0b0101_1010_0011_1100),
        },
        SeedingCase {
            name: "6_teams",
            remaining: Set::from(0x0333),
            diffs: [2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 3, -3, 0, 0],
            opponents: opponent_fixture(0b0110_1001_1010_0101),
        },
        SeedingCase {
            name: "4_teams",
            remaining: Set::from(0x1248),
            diffs: [3, -3, 2, -2, 1, -1, 0, 0, -1, 1, -2, 2, -3, 3, 0, -3],
            opponents: opponent_fixture(0b1001_0001_0110_1010),
        },
    ]
}

fn bench_seeding(c: &mut Criterion) {
    let mut group = c.benchmark_group("seeding");

    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));

    for case in &seeding_cases() {
        group.bench_with_input(BenchmarkId::new("scalar", case.name), case, |b, case| {
            b.iter(|| {
                black_box(seeding::scalar_impl(
                    black_box(case.remaining),
                    black_box(&case.diffs),
                    black_box(&case.opponents),
                ));
            });
        });

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("sse2") {
            group.bench_with_input(BenchmarkId::new("sse2", case.name), case, |b, case| {
                b.iter(|| {
                    black_box(unsafe {
                        seeding::x86_64::sse2_impl(
                            black_box(case.remaining),
                            black_box(&case.diffs),
                            black_box(&case.opponents),
                        )
                    });
                });
            });
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            group.bench_with_input(BenchmarkId::new("avx2", case.name), case, |b, case| {
                b.iter(|| {
                    black_box(unsafe {
                        seeding::x86_64::avx2_impl(
                            black_box(case.remaining),
                            black_box(&case.diffs),
                            black_box(&case.opponents),
                        )
                    });
                });
            });
        }
    }

    group.finish();
}

struct ProbabilitiesCase {
    name: &'static str,
    ratings: [Rating; 16],
    sigma: f32,
}

fn ratings(values: [u16; 16]) -> [Rating; 16] {
    values.map(Rating::new)
}

fn probabilities_cases() -> [ProbabilitiesCase; 3] {
    [
        ProbabilitiesCase {
            name: "spread",
            ratings: ratings([
                1_100, 1_180, 1_260, 1_340, 1_420, 1_500, 1_580, 1_660, 1_740, 1_820, 1_900, 1_980,
                2_060, 2_140, 2_220, 2_300,
            ]),
            sigma: 800.0,
        },
        ProbabilitiesCase {
            name: "mixed",
            ratings: ratings([
                1_850, 1_120, 2_040, 1_530, 1_760, 2_270, 1_390, 1_970, 1_210, 2_130, 1_680, 1_470,
                2_360, 1_590, 1_910, 1_300,
            ]),
            sigma: 800.0,
        },
        ProbabilitiesCase {
            name: "close",
            ratings: ratings([
                1_600, 1_600, 1_610, 1_610, 1_620, 1_620, 1_630, 1_630, 1_640, 1_640, 1_650, 1_650,
                1_660, 1_660, 1_670, 1_670,
            ]),
            sigma: 400.0,
        },
    ]
}

fn bench_probabilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("probabilities");

    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));

    for case in &probabilities_cases() {
        group.bench_with_input(BenchmarkId::new("scalar", case.name), case, |b, case| {
            b.iter(|| {
                black_box(probabilities::scalar_impl(
                    black_box(case.ratings),
                    black_box(case.sigma),
                ));
            });
        });

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("sse2") {
            group.bench_with_input(BenchmarkId::new("sse2", case.name), case, |b, case| {
                b.iter(|| {
                    black_box(unsafe {
                        probabilities::x86_64::sse2_impl(
                            black_box(case.ratings),
                            black_box(case.sigma),
                        )
                    });
                });
            });
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            group.bench_with_input(BenchmarkId::new("avx2", case.name), case, |b, case| {
                b.iter(|| {
                    black_box(unsafe {
                        probabilities::x86_64::avx2_impl(
                            black_box(case.ratings),
                            black_box(case.sigma),
                        )
                    });
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simulation,
    bench_seeding,
    bench_probabilities
);
criterion_main!(benches);
