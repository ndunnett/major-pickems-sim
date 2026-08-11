//! Optimised probability-matrix implementations for x86_64.

#![allow(clippy::wildcard_imports)]

use std::{arch::x86_64::*, f32::consts::LOG2_10, mem::MaybeUninit};

use crate::datatypes::{Rating, Sigma};

use super::exp2_constants;

/// Precalculate BO1 and BO3 win-probability matrices.
///
/// The result is `[probabilities_bo1, probabilities_bo3]`, and each matrix is
/// indexed by `[team_a][team_b]`.
#[must_use]
#[inline]
pub fn calculate_probabilities(ratings: [Rating; 16], sigma: Sigma) -> [[[f32; 16]; 16]; 2] {
    if is_x86_feature_detected!("avx2") {
        unsafe { avx2_impl(ratings, sigma) }
    } else {
        // Safety: SSE2 is enabled on x86_64 architectures by default
        unsafe { super::x86::sse2_impl(ratings, sigma) }
    }
}

/// Precalculate BO1 and BO3 win-probability matrices using AVX2.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support AVX2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("avx2")` or an equivalent guarantee.
#[target_feature(enable = "avx2")]
#[must_use]
pub fn avx2_impl(ratings: [Rating; 16], sigma: Sigma) -> [[[f32; 16]; 16]; 2] {
    let u = _mm256_set1_ps(LOG2_10 / sigma.to_f32());
    let mut probabilities = MaybeUninit::<[[[f32; 16]; 16]; 2]>::uninit();
    let probabilities_bo1 = probabilities.as_mut_ptr().cast::<f32>();
    let probabilities_bo3 = unsafe { probabilities_bo1.add(consts::MATRIX_LEN) };
    let ratings = ratings.map(Rating::to_f32);
    let ratings_vecs = unsafe {
        [
            _mm256_loadu_ps(ratings.as_ptr()),
            _mm256_loadu_ps(ratings.as_ptr().add(consts::LANES)),
        ]
    };

    // Compute both diagonal blocks densely.
    for (row, &ra) in ratings.iter().take(consts::LANES).enumerate() {
        let p = probability(_mm256_set1_ps(ra), ratings_vecs[0], u);
        let offset = row * consts::TEAM_COUNT;
        unsafe { _mm256_storeu_ps(probabilities_bo1.add(offset), p) };
    }

    for (row, &ra) in ratings.iter().enumerate().skip(consts::LANES) {
        let p = probability(_mm256_set1_ps(ra), ratings_vecs[1], u);
        let offset = row * consts::TEAM_COUNT + consts::LANES;
        unsafe { _mm256_storeu_ps(probabilities_bo1.add(offset), p) };
    }

    // Compute the upper-right block once and mirror its complement into the
    // lower-left block.
    let mut p = [_mm256_setzero_ps(); consts::LANES];

    for (row, value) in p.iter_mut().enumerate() {
        *value = probability(_mm256_set1_ps(ratings[row]), ratings_vecs[1], u);
        let offset = row * consts::TEAM_COUNT + consts::LANES;
        unsafe { _mm256_storeu_ps(probabilities_bo1.add(offset), *value) };
    }

    for (row, value) in transpose(p).into_iter().enumerate() {
        let value = _mm256_sub_ps(consts::ONE, value);
        let offset = (row + consts::LANES) * consts::TEAM_COUNT;
        unsafe { _mm256_storeu_ps(probabilities_bo1.add(offset), value) };
    }

    // Convert each BO1 lane into the corresponding BO3 series probability.
    for offset in (0..consts::MATRIX_LEN).step_by(consts::LANES) {
        // a = P * P
        // x = 3 - 2 * P
        // Q = a * x
        let p = unsafe { _mm256_loadu_ps(probabilities_bo1.add(offset)) };
        let a = _mm256_mul_ps(p, p);
        let x = _mm256_sub_ps(consts::THREE, _mm256_mul_ps(consts::TWO, p));
        let q = _mm256_mul_ps(a, x);
        unsafe { _mm256_storeu_ps(probabilities_bo3.add(offset), q) };
    }

    // Safety: the loops above initialise all 512 values in the result.
    unsafe { probabilities.assume_init() }
}

/// Compute eight BO1 probabilities for one team against eight opponents.
#[inline]
#[target_feature(enable = "avx2")]
fn probability(ra: __m256, rb: __m256, u: __m256) -> __m256 {
    // u = log2(10) / sigma
    // v = (Rb - Ra) * u
    // w = 2^v
    // P = 1 / (1 + w)
    let v = _mm256_mul_ps(u, _mm256_sub_ps(rb, ra));
    let w = exp2(v);
    _mm256_div_ps(consts::ONE, _mm256_add_ps(consts::ONE, w))
}

/// Transpose an 8x8 matrix of packed `f32` lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose(v: [__m256; 8]) -> [__m256; 8] {
    // Build the transpose in three stages: pairwise interleaves, 4-lane groups
    // inside each 128-bit half, then cross-half permutes. The result lets the
    // mirrored BO1 block be stored as contiguous rows.
    let pairs = [
        _mm256_unpacklo_ps(v[0], v[1]),
        _mm256_unpackhi_ps(v[0], v[1]),
        _mm256_unpacklo_ps(v[2], v[3]),
        _mm256_unpackhi_ps(v[2], v[3]),
        _mm256_unpacklo_ps(v[4], v[5]),
        _mm256_unpackhi_ps(v[4], v[5]),
        _mm256_unpacklo_ps(v[6], v[7]),
        _mm256_unpackhi_ps(v[6], v[7]),
    ];

    let quads = [
        _mm256_shuffle_ps::<{ consts::SHUFFLE_LO }>(pairs[0], pairs[2]),
        _mm256_shuffle_ps::<{ consts::SHUFFLE_HI }>(pairs[0], pairs[2]),
        _mm256_shuffle_ps::<{ consts::SHUFFLE_LO }>(pairs[1], pairs[3]),
        _mm256_shuffle_ps::<{ consts::SHUFFLE_HI }>(pairs[1], pairs[3]),
        _mm256_shuffle_ps::<{ consts::SHUFFLE_LO }>(pairs[4], pairs[6]),
        _mm256_shuffle_ps::<{ consts::SHUFFLE_HI }>(pairs[4], pairs[6]),
        _mm256_shuffle_ps::<{ consts::SHUFFLE_LO }>(pairs[5], pairs[7]),
        _mm256_shuffle_ps::<{ consts::SHUFFLE_HI }>(pairs[5], pairs[7]),
    ];

    [
        _mm256_permute2f128_ps::<{ consts::PERMUTE_LO }>(quads[0], quads[4]),
        _mm256_permute2f128_ps::<{ consts::PERMUTE_LO }>(quads[1], quads[5]),
        _mm256_permute2f128_ps::<{ consts::PERMUTE_LO }>(quads[2], quads[6]),
        _mm256_permute2f128_ps::<{ consts::PERMUTE_LO }>(quads[3], quads[7]),
        _mm256_permute2f128_ps::<{ consts::PERMUTE_HI }>(quads[0], quads[4]),
        _mm256_permute2f128_ps::<{ consts::PERMUTE_HI }>(quads[1], quads[5]),
        _mm256_permute2f128_ps::<{ consts::PERMUTE_HI }>(quads[2], quads[6]),
        _mm256_permute2f128_ps::<{ consts::PERMUTE_HI }>(quads[3], quads[7]),
    ]
}

/// Approximate `2^x` for eight packed `f32` lanes using AVX2.
///
/// The approximation splits each input into an integer power of two and a small
/// exponential remainder:
///
/// `2^x = 2^n * exp((x - n) * ln(2))`, where `n = floor(x + 0.5)`.
#[target_feature(enable = "avx2")]
fn exp2(x: __m256) -> __m256 {
    // Clamp x so the reconstructed 2^n stays within normal f32 exponent bits.
    let x = _mm256_max_ps(_mm256_min_ps(x, consts::INPUT_MAX), consts::INPUT_MIN);

    // Choose n = floor(x + 0.5), the nearest integer with ties toward +inf.
    let rounded = _mm256_add_ps(x, consts::HALF);

    // Convert with truncation toward zero.
    let truncated = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(rounded));

    // For negative non-integers, floor(v) = trunc(v) - 1.
    let floor_adj = _mm256_and_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(truncated, rounded), consts::ONE);
    let n = _mm256_sub_ps(truncated, floor_adj);

    // Reduce to z = (x - n) * ln(2), so 2^x = 2^n * exp(z).
    let z = _mm256_mul_ps(_mm256_sub_ps(x, n), consts::LN_2);

    // Approximate exp(z) as 1 + z + z^2 * P(z), with P evaluated in
    // independent pairs to shorten the multiplication dependency chain.
    let z2 = _mm256_mul_ps(z, z);
    let p01 = _mm256_add_ps(_mm256_mul_ps(consts::POLY[0], z), consts::POLY[1]);
    let p23 = _mm256_add_ps(_mm256_mul_ps(consts::POLY[2], z), consts::POLY[3]);
    let p45 = _mm256_add_ps(_mm256_mul_ps(consts::POLY[4], z), consts::POLY[5]);
    let p0123 = _mm256_add_ps(_mm256_mul_ps(p01, z2), p23);
    let polynomial = _mm256_add_ps(_mm256_mul_ps(p0123, z2), p45);
    let y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(polynomial, z2), z), consts::ONE);

    // Build 2^n directly as an IEEE-754 f32: exponent bits = n + bias, mantissa = 0.
    let exponent = _mm256_cvttps_epi32(n);
    let biased_exponent = _mm256_add_epi32(exponent, consts::EXPONENT_BIAS);
    let exponent_bits = _mm256_slli_epi32::<{ exp2_constants::MANTISSA_BITS }>(biased_exponent);
    let pow2n = _mm256_castsi256_ps(exponent_bits);

    // 2^x = exp(z) * 2^n.
    _mm256_mul_ps(y, pow2n)
}

mod consts {
    use std::mem::transmute;

    use super::*;

    const fn f32x8(data: [f32; 8]) -> __m256 {
        unsafe { transmute(data) }
    }

    const fn i32x8(data: [i32; 8]) -> __m256i {
        unsafe { transmute(data) }
    }

    pub const LANES: usize = 8;
    pub const TEAM_COUNT: usize = 16;
    pub const MATRIX_LEN: usize = TEAM_COUNT * TEAM_COUNT;
    pub const SHUFFLE_LO: i32 = 0x44;
    pub const SHUFFLE_HI: i32 = 0xEE;
    pub const PERMUTE_LO: i32 = 0x20;
    pub const PERMUTE_HI: i32 = 0x31;

    pub const HALF: __m256 = f32x8([0.5; _]);
    pub const ONE: __m256 = f32x8([1.0; _]);
    pub const TWO: __m256 = f32x8([2.0; _]);
    pub const THREE: __m256 = f32x8([3.0; _]);
    pub const LN_2: __m256 = f32x8([std::f32::consts::LN_2; _]);
    pub const INPUT_MIN: __m256 = f32x8([exp2_constants::INPUT_MIN; _]);
    pub const INPUT_MAX: __m256 = f32x8([exp2_constants::INPUT_MAX; _]);
    pub const EXPONENT_BIAS: __m256i = i32x8([exp2_constants::EXPONENT_BIAS; _]);

    pub const POLY: [__m256; 6] = [
        f32x8([exp2_constants::POLY_0; _]),
        f32x8([exp2_constants::POLY_1; _]),
        f32x8([exp2_constants::POLY_2; _]),
        f32x8([exp2_constants::POLY_3; _]),
        f32x8([exp2_constants::POLY_4; _]),
        f32x8([exp2_constants::POLY_5; _]),
    ];
}
