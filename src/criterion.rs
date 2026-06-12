use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use pickems::{
    datatypes::{Iterations, Rating, Set, Sigma},
    reporting::NullReport,
    simulation::{
        Simulation, probabilities,
        seeding::{self, PackedSeeding},
        sorting,
    },
};

fn bench_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation");

    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .bench_function("parallel", |b| {
            let sim = Simulation::dummy(Iterations::new(1_000_000));
            b.iter(|| sim.clone().run(NullReport));
        })
        .bench_function("single_thread", |b| {
            let sim = Simulation::dummy(Iterations::new(50_000));
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
    let cases = seeding_cases();
    let mut group = c.benchmark_group("seeding");

    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("scalar", case.name), case, |b, case| {
            b.iter(|| {
                black_box(seeding::scalar_impl(
                    black_box(case.remaining),
                    black_box(&case.diffs),
                    black_box(&case.opponents),
                ));
            });
        });
    }

    // Runtime of SIMD implementations does not dependent on arguments.
    let case = &cases[0];

    #[cfg(target_feature = "neon")]
    group.bench_with_input(BenchmarkId::new("neon", case.name), case, |b, case| {
        b.iter(|| {
            black_box(unsafe {
                seeding::aarch64::neon_impl(
                    black_box(case.remaining),
                    black_box(&case.diffs),
                    black_box(&case.opponents),
                )
            });
        });
    });

    #[cfg(target_feature = "sse2")]
    group.bench_with_input(BenchmarkId::new("sse2", case.name), case, |b, case| {
        b.iter(|| {
            black_box(unsafe {
                seeding::x86::sse2_impl(
                    black_box(case.remaining),
                    black_box(&case.diffs),
                    black_box(&case.opponents),
                )
            });
        });
    });

    #[cfg(target_feature = "avx2")]
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

    group.finish();
}

struct SortingCase {
    name: &'static str,
    seeding: PackedSeeding,
    len: usize,
}

fn sorting_cases() -> [SortingCase; 5] {
    [
        SortingCase {
            name: "16_items",
            seeding: PackedSeeding::from([
                0x1020, 0x0010, 0x4010, 0x0001, 0x2400, 0x1000, 0x0100, 0x0000, 0x2200, 0x0200,
                0x2100, 0x0030, 0x1300, 0x1200, 0x1100, 0x0110,
            ]),
            len: 16,
        },
        SortingCase {
            name: "12_items",
            seeding: PackedSeeding::from([
                0x1020,
                0x0010,
                0x4010,
                0x0001,
                0x2400,
                0x1000,
                0x0100,
                0x0000,
                0x2200,
                0x0200,
                0x2100,
                0x0030,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
            ]),
            len: 12,
        },
        SortingCase {
            name: "8_items",
            seeding: PackedSeeding::from([
                0x1020,
                0x0010,
                0x4010,
                0x0001,
                0x2400,
                0x1000,
                0x0100,
                0x0000,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
            ]),
            len: 8,
        },
        SortingCase {
            name: "6_items",
            seeding: PackedSeeding::from([
                u16::MAX,
                0x0010,
                0x4010,
                u16::MAX,
                0x2400,
                0x1000,
                u16::MAX,
                0x0000,
                u16::MAX,
                u16::MAX,
                0x2100,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
            ]),
            len: 6,
        },
        SortingCase {
            name: "4_items",
            seeding: PackedSeeding::from([
                u16::MAX,
                u16::MAX,
                0x4010,
                u16::MAX,
                0x2400,
                u16::MAX,
                u16::MAX,
                0x0000,
                u16::MAX,
                u16::MAX,
                0x2100,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
                u16::MAX,
            ]),
            len: 4,
        },
    ]
}

fn bench_sorting(c: &mut Criterion) {
    let cases = sorting_cases();
    let mut group = c.benchmark_group("sorting");

    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("scalar", case.name), case, |b, case| {
            b.iter(|| {
                let seeding = black_box(case.seeding);
                black_box(seeding.sort_strip(black_box(case.len)));
            });
        });
    }

    // Runtime of SIMD implementations does not dependent on arguments.
    let case = &cases[0];

    #[cfg(target_feature = "neon")]
    group.bench_with_input(BenchmarkId::new("neon", case.name), case, |b, case| {
        b.iter(|| {
            black_box(unsafe {
                sorting::aarch64::sort_strip_neon(black_box(case.seeding), black_box(case.len))
            });
        });
    });

    #[cfg(target_feature = "sse2")]
    group.bench_with_input(BenchmarkId::new("sse2", case.name), case, |b, case| {
        b.iter(|| {
            black_box(unsafe {
                sorting::x86::sort_strip_sse2(black_box(case.seeding), black_box(case.len))
            });
        });
    });

    #[cfg(target_feature = "avx2")]
    group.bench_with_input(BenchmarkId::new("avx2", case.name), case, |b, case| {
        b.iter(|| {
            black_box(unsafe {
                sorting::x86_64::sort_strip_avx2(black_box(case.seeding), black_box(case.len))
            });
        });
    });

    group.finish();
}

fn bench_probabilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("probabilities");

    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));

    let case = &(
        [
            1_850, 1_120, 2_040, 1_530, 1_760, 2_270, 1_390, 1_970, 1_210, 2_130, 1_680, 1_470,
            2_360, 1_590, 1_910, 1_300,
        ]
        .map(Rating::new),
        Sigma::new(800.0),
    );

    group.bench_with_input("scalar", case, |b, &(ratings, sigma)| {
        b.iter(|| {
            black_box(probabilities::scalar_impl(
                black_box(ratings),
                black_box(sigma),
            ));
        });
    });

    #[cfg(target_feature = "neon")]
    group.bench_with_input("neon", case, |b, &(ratings, sigma)| {
        b.iter(|| {
            black_box(unsafe {
                probabilities::aarch64::neon_impl(black_box(ratings), black_box(sigma))
            });
        });
    });

    #[cfg(target_feature = "sse2")]
    group.bench_with_input("sse2", case, |b, &(ratings, sigma)| {
        b.iter(|| {
            black_box(unsafe {
                probabilities::x86::sse2_impl(black_box(ratings), black_box(sigma))
            });
        });
    });

    #[cfg(target_feature = "avx2")]
    group.bench_with_input("avx2", case, |b, &(ratings, sigma)| {
        b.iter(|| {
            black_box(unsafe {
                probabilities::x86_64::avx2_impl(black_box(ratings), black_box(sigma))
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_simulation,
    bench_seeding,
    bench_sorting,
    bench_probabilities
);
criterion_main!(benches);
