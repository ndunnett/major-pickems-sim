use std::hint::black_box;

use criterion::Criterion;

use pickems::{
    datatypes::{Rating, Sigma},
    simulation::probabilities,
};

fn case() -> ([Rating; 16], Sigma) {
    (
        [
            1_850, 1_120, 2_040, 1_530, 1_760, 2_270, 1_390, 1_970, 1_210, 2_130, 1_680, 1_470,
            2_360, 1_590, 1_910, 1_300,
        ]
        .map(Rating::new),
        Sigma::new(800.0),
    )
}

pub fn bench(c: &mut Criterion) {
    let case = case();
    let mut group = c.benchmark_group("probabilities");

    group.bench_with_input("scalar", &case, |b, &(ratings, sigma)| {
        b.iter(|| {
            black_box(probabilities::scalar_impl(
                black_box(ratings),
                black_box(sigma),
            ));
        });
    });

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        group.bench_with_input("neon", &case, |b, &(ratings, sigma)| {
            b.iter(|| {
                black_box(unsafe {
                    probabilities::aarch64::neon_impl(black_box(ratings), black_box(sigma))
                });
            });
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        group.bench_with_input("sse2", &case, |b, &(ratings, sigma)| {
            b.iter(|| {
                black_box(unsafe {
                    probabilities::x86::sse2_impl(black_box(ratings), black_box(sigma))
                });
            });
        });
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        group.bench_with_input("avx2", &case, |b, &(ratings, sigma)| {
            b.iter(|| {
                black_box(unsafe {
                    probabilities::x86_64::avx2_impl(black_box(ratings), black_box(sigma))
                });
            });
        });
    }

    group.finish();
}
