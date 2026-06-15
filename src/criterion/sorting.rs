use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use pickems::simulation::{seeding::PackedSeeding, sorting};

struct Case {
    name: &'static str,
    seeding: PackedSeeding,
    len: usize,
}

fn cases() -> [Case; 5] {
    [
        Case {
            name: "16_items",
            seeding: PackedSeeding::from([
                0x1020, 0x0010, 0x4010, 0x0001, 0x2400, 0x1000, 0x0100, 0x0000, 0x2200, 0x0200,
                0x2100, 0x0030, 0x1300, 0x1200, 0x1100, 0x0110,
            ]),
            len: 16,
        },
        Case {
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
        Case {
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
        Case {
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
        Case {
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

pub fn bench(c: &mut Criterion) {
    let cases = cases();
    let mut group = c.benchmark_group("sorting");

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("scalar", case.name), case, |b, case| {
            b.iter(|| {
                let seeding = black_box(case.seeding);
                black_box(seeding.sort_strip(black_box(case.len)));
            });
        });
    }

    // Runtime of SIMD implementations does not depend on the number of active items.
    let case = &cases[0];

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        group.bench_with_input(BenchmarkId::new("neon", case.name), case, |b, case| {
            b.iter(|| {
                black_box(unsafe {
                    sorting::aarch64::sort_strip_neon(black_box(case.seeding), black_box(case.len))
                });
            });
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        group.bench_with_input(BenchmarkId::new("sse2", case.name), case, |b, case| {
            b.iter(|| {
                black_box(unsafe {
                    sorting::x86::sort_strip_sse2(black_box(case.seeding), black_box(case.len))
                });
            });
        });
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        group.bench_with_input(BenchmarkId::new("avx2", case.name), case, |b, case| {
            b.iter(|| {
                black_box(unsafe {
                    sorting::x86_64::sort_strip_avx2(black_box(case.seeding), black_box(case.len))
                });
            });
        });
    }

    group.finish();
}
