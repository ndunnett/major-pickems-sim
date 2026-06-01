//! Optimised probability-matrix implementations for x86.

#![allow(clippy::wildcard_imports)]

cfg_select! {
    target_arch = "x86" => {
        use std::arch::x86::*;
    }
    target_arch = "x86_64" => {
        use std::arch::x86_64::*;
    }
}

use std::f32::consts::LN_2;

use crate::datatypes::{Rating, Sigma};

use super::exp2_constants::*;

/// Precalculate BO1 and BO3 win-probability matrices.
///
/// The result is `[probabilities_bo1, probabilities_bo3]`, and each matrix is
/// indexed by `[team_a][team_b]`.
#[cfg(target_arch = "x86")]
#[must_use]
#[inline]
pub fn calculate_probabilities(ratings: [Rating; 16], sigma: Sigma) -> [[[f32; 16]; 16]; 2] {
    if is_x86_feature_detected!("sse2") {
        unsafe { super::x86_64::sse2_impl(ratings, sigma) }
    } else {
        super::scalar_impl(ratings, sigma)
    }
}

/// Precalculate BO1 and BO3 win-probability matrices using SSE2.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support SSE2 instructions.
#[target_feature(enable = "sse2")]
#[must_use]
pub unsafe fn sse2_impl(ratings: [Rating; 16], sigma: Sigma) -> [[[f32; 16]; 16]; 2] {
    let mut r = [f32::NAN; 16];

    for i in 0..16 {
        r[i] = ratings[i].to_f32();
    }

    // u = log2(10) / sigma
    // v = (Rb - Ra) * u
    // w = 2^v
    // P = 1 / (1 + w)
    let u = _mm_set1_ps(std::f32::consts::LOG2_10 / sigma.to_f32());
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
    let x = _mm_min_ps(x, _mm_set1_ps(INPUT_MAX));
    let x = _mm_max_ps(x, _mm_set1_ps(INPUT_MIN));

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
    let mut y = _mm_set1_ps(POLY_0);
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(POLY_1));
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(POLY_2));
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(POLY_3));
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(POLY_4));
    y = _mm_add_ps(_mm_mul_ps(y, z), _mm_set1_ps(POLY_5));
    y = _mm_mul_ps(y, z);
    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, z);
    y = _mm_add_ps(y, one);

    // Build 2^n directly as an IEEE-754 f32: exponent bits = n + bias, mantissa = 0.
    let exponent = _mm_cvttps_epi32(n);
    let biased_exponent = _mm_add_epi32(exponent, _mm_set1_epi32(EXPONENT_BIAS));
    let exponent_bits = _mm_slli_epi32(biased_exponent, MANTISSA_BITS);
    let pow2n = _mm_castsi128_ps(exponent_bits);

    // 2^x = exp(z) * 2^n.
    _mm_mul_ps(y, pow2n)
}
