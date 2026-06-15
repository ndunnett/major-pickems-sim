use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use pickems::{datatypes::Set, simulation::seeding};

struct Case {
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

fn cases() -> [Case; 5] {
    [
        Case {
            name: "16_teams",
            remaining: Set::from(0xFFFF),
            diffs: [0; 16],
            opponents: opponent_fixture(0b0001_0010_1010_0101),
        },
        Case {
            name: "12_teams",
            remaining: Set::from(0x0FFF),
            diffs: [3, 3, 2, 2, 2, 1, 1, 1, -1, -1, -1, -2, -2, -2, -3, -3],
            opponents: opponent_fixture(0b0011_0101_1000_1110),
        },
        Case {
            name: "8_teams",
            remaining: Set::from(0x5555),
            diffs: [1, 1, 1, 1, -1, -1, -1, -1, 2, 2, -2, -2, 3, -3, 0, 0],
            opponents: opponent_fixture(0b0101_1010_0011_1100),
        },
        Case {
            name: "6_teams",
            remaining: Set::from(0x0333),
            diffs: [2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 3, -3, 0, 0],
            opponents: opponent_fixture(0b0110_1001_1010_0101),
        },
        Case {
            name: "4_teams",
            remaining: Set::from(0x1248),
            diffs: [3, -3, 2, -2, 1, -1, 0, 0, -1, 1, -2, 2, -3, 3, 0, -3],
            opponents: opponent_fixture(0b1001_0001_0110_1010),
        },
    ]
}

pub fn bench(c: &mut Criterion) {
    let cases = cases();
    let mut group = c.benchmark_group("seeding");

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

    // Runtime of SIMD implementations does not depend on the number of remaining teams.
    let case = &cases[0];

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
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
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
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
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
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

    group.finish();
}
