//! Optimised probability-matrix implementations for x86 and x86_64.

cfg_select! {
    target_arch = "x86" => {
        #[allow(clippy::wildcard_imports)]
        use std::arch::x86::*;
    }
    target_arch = "x86_64" => {
        #[allow(clippy::wildcard_imports)]
        use std::arch::x86_64::*;
    }
}

use std::f32::consts::LN_2;

use crate::datatypes::Rating;

const EXP2_INPUT_MIN: f32 = -126.0;
const EXP2_INPUT_MAX: f32 = 127.0;
const F32_EXPONENT_BIAS: i32 = 0x7f;
const F32_MANTISSA_BITS: i32 = 23;

const EXP_POLY_0: f32 = 0.000_198_756_91;
const EXP_POLY_1: f32 = 0.001_398_199_9;
const EXP_POLY_2: f32 = 0.008_333_452;
const EXP_POLY_3: f32 = 0.041_665_796;
const EXP_POLY_4: f32 = 0.166_666_66;
const EXP_POLY_5: f32 = 0.500_000_06;

/// Precalculate BO1 and BO3 win-probability matrices.
///
/// The result is `[probabilities_bo1, probabilities_bo3]`, and each matrix is
/// indexed by `[team_a][team_b]`.
#[must_use]
#[inline]
pub fn calculate_probabilities(ratings: [Rating; 16], sigma: f32) -> [[[f32; 16]; 16]; 2] {
    if is_x86_feature_detected!("avx2") {
        unsafe { avx2_impl(ratings, sigma) }
    } else {
        // SSE2 is enabled on x86 architectures by default
        unsafe { sse2_impl(ratings, sigma) }
    }
}

/// Precalculate BO1 and BO3 win-probability matrices using SSE2 lanes.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support SSE2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("sse2")` or an equivalent guarantee.
#[target_feature(enable = "sse2")]
#[must_use]
pub unsafe fn sse2_impl(ratings: [Rating; 16], sigma: f32) -> [[[f32; 16]; 16]; 2] {
    let mut r = [f32::NAN; 16];

    for i in 0..16 {
        r[i] = ratings[i].to_f32();
    }

    // u = log2(10) / sigma
    // v = (Rb - Ra) * u
    // w = 2^v
    // P = 1 / (1 + w)
    let u = _mm_set1_ps(std::f32::consts::LOG2_10 / sigma);
    let one = _mm_set1_ps(1.0);
    let mut probabilities_bo1 = [[f32::NAN; 16]; 16];

    for i in 0..16 {
        let ra = _mm_set1_ps(r[i]);

        for j in (0..16).step_by(4) {
            unsafe {
                // Load four potential opponents and compute row `i` of the BO1 matrix.
                let rb = _mm_loadu_ps(r.as_ptr().add(j));
                let v = _mm_mul_ps(u, _mm_sub_ps(rb, ra));
                let w = exp2_ps_sse2(v);
                let p = _mm_div_ps(one, _mm_add_ps(one, w));
                _mm_storeu_ps(probabilities_bo1[i].as_mut_ptr().add(j), p);
            }
        }
    }

    // a = P * P
    // b = 1 - P
    // Q = 2 * a * b + a
    let two = _mm_set1_ps(2.0);
    let mut probabilities_bo3 = [[f32::NAN; 16]; 16];

    for i in 0..16 {
        for j in (0..16).step_by(4) {
            unsafe {
                // Convert each BO1 lane into the corresponding BO3 series probability.
                let p = _mm_loadu_ps(probabilities_bo1[i].as_ptr().add(j));
                let a = _mm_mul_ps(p, p);
                let b = _mm_sub_ps(one, p);
                let q = _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(a, b)), a);
                _mm_storeu_ps(probabilities_bo3[i].as_mut_ptr().add(j), q);
            }
        }
    }

    [probabilities_bo1, probabilities_bo3]
}

/// Precalculate BO1 and BO3 win-probability matrices using AVX2 lanes.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support AVX2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("avx2")` or an equivalent guarantee.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn avx2_impl(ratings: [Rating; 16], sigma: f32) -> [[[f32; 16]; 16]; 2] {
    let mut r = [f32::NAN; 16];

    for i in 0..16 {
        r[i] = ratings[i].to_f32();
    }

    // u = log2(10) / sigma
    // v = (Rb - Ra) * u
    // w = 2^v
    // P = 1 / (1 + w)
    let u = _mm256_set1_ps(std::f32::consts::LOG2_10 / sigma);
    let one = _mm256_set1_ps(1.0);
    let mut probabilities_bo1 = [[f32::NAN; 16]; 16];

    for i in 0..16 {
        let ra = _mm256_set1_ps(r[i]);

        for j in (0..16).step_by(8) {
            unsafe {
                // Load eight potential opponents and compute row `i` of the BO1 matrix.
                let rb = _mm256_loadu_ps(r.as_ptr().add(j));
                let v = _mm256_mul_ps(u, _mm256_sub_ps(rb, ra));
                let w = exp2_ps_avx2(v);
                let p = _mm256_div_ps(one, _mm256_add_ps(one, w));
                _mm256_storeu_ps(probabilities_bo1[i].as_mut_ptr().add(j), p);
            }
        }
    }

    // a = P * P
    // b = 1 - P
    // Q = 2 * a * b + a
    let two = _mm256_set1_ps(2.0);
    let mut probabilities_bo3 = [[f32::NAN; 16]; 16];

    for i in 0..16 {
        for j in (0..16).step_by(8) {
            unsafe {
                // Convert each BO1 lane into the corresponding BO3 series probability.
                let p = _mm256_loadu_ps(probabilities_bo1[i].as_ptr().add(j));
                let a = _mm256_mul_ps(p, p);
                let b = _mm256_sub_ps(one, p);
                let q = _mm256_add_ps(_mm256_mul_ps(two, _mm256_mul_ps(a, b)), a);
                _mm256_storeu_ps(probabilities_bo3[i].as_mut_ptr().add(j), q);
            }
        }
    }

    [probabilities_bo1, probabilities_bo3]
}

/// Approximate `2^x` for four packed `f32` lanes using SSE2.
///
/// The approximation splits each input into an integer power of two and a small
/// exponential remainder:
///
/// `2^x = 2^n * exp((x - n) * ln(2))`, where `n = floor(x + 0.5)`.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support SSE2 instructions.
#[target_feature(enable = "sse2")]
unsafe fn exp2_ps_sse2(x: __m128) -> __m128 {
    let one = _mm_set1_ps(1.0);

    // Clamp x so the reconstructed 2^n stays within normal f32 exponent bits.
    let x = _mm_min_ps(x, _mm_set1_ps(EXP2_INPUT_MAX));
    let x = _mm_max_ps(x, _mm_set1_ps(EXP2_INPUT_MIN));

    // Choose n = floor(x + 0.5), the nearest integer with ties toward +inf.
    let rounded = _mm_add_ps(x, _mm_set1_ps(0.5));
    // Convert with truncation toward zero.
    let truncated = _mm_cvttps_epi32(rounded);
    let truncated = _mm_cvtepi32_ps(truncated);
    // For negative non-integers, floor(v) = trunc(v) - 1.
    let floor_correction = _mm_and_ps(_mm_cmpgt_ps(truncated, rounded), one);
    let n = _mm_sub_ps(truncated, floor_correction);

    // Reduce to z = (x - n) * ln(2), so 2^x = 2^n * exp(z).
    let z = _mm_sub_ps(x, n);
    let z = _mm_mul_ps(z, _mm_set1_ps(LN_2));

    // Approximate exp(z) as 1 + z + z^2 * P(z), with P evaluated by Horner's rule.
    let mut y = _mm_set1_ps(EXP_POLY_0);
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(EXP_POLY_1));
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(EXP_POLY_2));
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(EXP_POLY_3));
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(EXP_POLY_4));
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(EXP_POLY_5));
    y = _mm_mul_ps(y, z);
    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, z);
    y = _mm_add_ps(y, one);

    // Build 2^n directly as an IEEE-754 f32: exponent bits = n + bias, mantissa = 0.
    let exponent = _mm_cvttps_epi32(n);
    let biased_exponent = _mm_add_epi32(exponent, _mm_set1_epi32(F32_EXPONENT_BIAS));
    let exponent_bits = _mm_slli_epi32(biased_exponent, F32_MANTISSA_BITS);
    let pow2n = _mm_castsi128_ps(exponent_bits);

    // 2^x = exp(z) * 2^n.
    _mm_mul_ps(y, pow2n)
}

/// Approximate `2^x` for eight packed `f32` lanes using AVX2.
///
/// The approximation splits each input into an integer power of two and a small
/// exponential remainder:
///
/// `2^x = 2^n * exp((x - n) * ln(2))`, where `n = floor(x + 0.5)`.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support AVX2 instructions.
#[target_feature(enable = "avx2")]
unsafe fn exp2_ps_avx2(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);

    // Clamp x so the reconstructed 2^n stays within normal f32 exponent bits.
    let x = _mm256_min_ps(x, _mm256_set1_ps(EXP2_INPUT_MAX));
    let x = _mm256_max_ps(x, _mm256_set1_ps(EXP2_INPUT_MIN));

    // Choose n = floor(x + 0.5), the nearest integer with ties toward +inf.
    let rounded = _mm256_add_ps(x, _mm256_set1_ps(0.5));
    // Convert with truncation toward zero.
    let truncated = _mm256_cvttps_epi32(rounded);
    let truncated = _mm256_cvtepi32_ps(truncated);
    // For negative non-integers, floor(v) = trunc(v) - 1.
    let floor_correction = _mm256_and_ps(_mm256_cmp_ps(truncated, rounded, _CMP_GT_OQ), one);
    let n = _mm256_sub_ps(truncated, floor_correction);

    // Reduce to z = (x - n) * ln(2), so 2^x = 2^n * exp(z).
    let z = _mm256_sub_ps(x, n);
    let z = _mm256_mul_ps(z, _mm256_set1_ps(LN_2));

    // Approximate exp(z) as 1 + z + z^2 * P(z), with P evaluated by Horner's rule.
    let mut y = _mm256_set1_ps(EXP_POLY_0);
    y = _mm256_add_ps(_mm256_mul_ps(y, z), _mm256_set1_ps(EXP_POLY_1));
    y = _mm256_add_ps(_mm256_mul_ps(y, z), _mm256_set1_ps(EXP_POLY_2));
    y = _mm256_add_ps(_mm256_mul_ps(y, z), _mm256_set1_ps(EXP_POLY_3));
    y = _mm256_add_ps(_mm256_mul_ps(y, z), _mm256_set1_ps(EXP_POLY_4));
    y = _mm256_add_ps(_mm256_mul_ps(y, z), _mm256_set1_ps(EXP_POLY_5));
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, z);
    y = _mm256_add_ps(y, one);

    // Build 2^n directly as an IEEE-754 f32: exponent bits = n + bias, mantissa = 0.
    let exponent = _mm256_cvttps_epi32(n);
    let biased_exponent = _mm256_add_epi32(exponent, _mm256_set1_epi32(F32_EXPONENT_BIAS));
    let exponent_bits = _mm256_slli_epi32(biased_exponent, F32_MANTISSA_BITS);
    let pow2n = _mm256_castsi256_ps(exponent_bits);

    // 2^x = exp(z) * 2^n.
    _mm256_mul_ps(y, pow2n)
}

#[cfg(test)]
mod tests {
    use super::{super::scalar_impl, Rating};

    const TOLERANCE: f32 = 1e-5;
    const SIGMAS: [f32; 2] = [800.0, 400.0];
    const RATING_FIXTURES: [[u16; 16]; 3] = [
        [
            1_100, 1_180, 1_260, 1_340, 1_420, 1_500, 1_580, 1_660, 1_740, 1_820, 1_900, 1_980,
            2_060, 2_140, 2_220, 2_300,
        ],
        [
            1_850, 1_120, 2_040, 1_530, 1_760, 2_270, 1_390, 1_970, 1_210, 2_130, 1_680, 1_470,
            2_360, 1_590, 1_910, 1_300,
        ],
        [
            1_600, 1_600, 1_610, 1_610, 1_620, 1_620, 1_630, 1_630, 1_640, 1_640, 1_650, 1_650,
            1_660, 1_660, 1_670, 1_670,
        ],
    ];

    type ImplFn = unsafe fn([Rating; 16], f32) -> [[[f32; 16]; 16]; 2];

    fn assert_matches_scalar(implementation: &str, func: ImplFn) {
        for fixture in RATING_FIXTURES {
            let ratings = fixture.map(Rating::new);

            for sigma in SIGMAS {
                let expected = scalar_impl(ratings, sigma);
                let actual = unsafe { func(ratings, sigma) };

                for matrix in 0..2 {
                    for row in 0..16 {
                        for column in 0..16 {
                            let expected = expected[matrix][row][column];
                            let actual = actual[matrix][row][column];
                            let difference = (expected - actual).abs();

                            assert!(
                                difference <= TOLERANCE,
                                "{implementation} differs from scalar at matrix {matrix}, row {row}, column {column}: expected {expected}, actual {actual}, difference {difference}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(target_feature = "sse2")]
    fn sse2() {
        assert_matches_scalar("sse2", super::sse2_impl);
    }

    #[test]
    #[cfg(target_feature = "avx2")]
    fn avx2() {
        assert_matches_scalar("avx2", super::avx2_impl);
    }
}
