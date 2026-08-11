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

use std::mem::MaybeUninit;

use crate::datatypes::{Rating, Sigma};

use super::exp2_constants;

/// Precalculate BO1 and BO3 win-probability matrices.
///
/// The result is `[probabilities_bo1, probabilities_bo3]`, and each matrix is
/// indexed by `[team_a][team_b]`.
#[cfg(target_arch = "x86")]
#[must_use]
#[inline]
pub fn calculate_probabilities(ratings: [Rating; 16], sigma: Sigma) -> [[[f32; 16]; 16]; 2] {
    if is_x86_feature_detected!("sse2") {
        unsafe { sse2_impl(ratings, sigma) }
    } else {
        super::scalar_impl(ratings, sigma)
    }
}

/// Precalculate BO1 and BO3 win-probability matrices using SSE2.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support SSE2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("sse2")` or an equivalent guarantee.
#[target_feature(enable = "sse2")]
#[must_use]
pub fn sse2_impl(ratings: [Rating; 16], sigma: Sigma) -> [[[f32; 16]; 16]; 2] {
    let ratings = ratings.map(Rating::to_f32);

    // u = log2(10) / sigma
    // v = (Rb - Ra) * u
    // w = 2^v
    // P = 1 / (1 + w)
    let u = _mm_set1_ps(std::f32::consts::LOG2_10 / sigma.to_f32());
    let mut probabilities = MaybeUninit::<[[[f32; 16]; 16]; 2]>::uninit();
    let probabilities_bo1 = probabilities.as_mut_ptr().cast::<f32>();
    let probabilities_bo3 = unsafe { probabilities_bo1.add(16 * 16) };

    for block in (0..16).step_by(4) {
        let rb = unsafe { _mm_loadu_ps(ratings.as_ptr().add(block)) };

        // Compute the diagonal block densely.
        for (row, &ra) in ratings.iter().enumerate().skip(block).take(4) {
            let p = probability(_mm_set1_ps(ra), rb, u, consts::ONE);
            unsafe {
                _mm_storeu_ps(probabilities_bo1.add(row * 16 + block), p);
            }
        }

        for column in (block + 4..16).step_by(4) {
            let rb = unsafe { _mm_loadu_ps(ratings.as_ptr().add(column)) };
            let mut p = [_mm_setzero_ps(); 4];

            for (lane, value) in p.iter_mut().enumerate() {
                *value = probability(_mm_set1_ps(ratings[block + lane]), rb, u, consts::ONE);
                unsafe {
                    _mm_storeu_ps(probabilities_bo1.add((block + lane) * 16 + column), *value);
                }
            }

            // P(B beats A) = 1 - P(A beats B). Transpose the block so each
            // mirrored row remains a contiguous store.
            for (lane, value) in transpose(p).into_iter().enumerate() {
                unsafe {
                    _mm_storeu_ps(
                        probabilities_bo1.add((column + lane) * 16 + block),
                        _mm_sub_ps(consts::ONE, value),
                    );
                }
            }
        }
    }

    // a = P * P
    // Q = a * (3 - 2 * P)
    for offset in (0..16 * 16).step_by(4) {
        // Convert each BO1 lane into the corresponding BO3 series probability.
        let p = unsafe { _mm_loadu_ps(probabilities_bo1.add(offset)) };
        let a = _mm_mul_ps(p, p);
        let q = _mm_mul_ps(a, _mm_sub_ps(consts::THREE, _mm_mul_ps(consts::TWO, p)));
        unsafe {
            _mm_storeu_ps(probabilities_bo3.add(offset), q);
        }
    }

    // Safety: the loops above initialise all 512 lanes in the result.
    unsafe { probabilities.assume_init() }
}

/// Compute four BO1 probabilities for one team against four opponents.
#[inline]
#[target_feature(enable = "sse2")]
fn probability(ra: __m128, rb: __m128, u: __m128, one: __m128) -> __m128 {
    let v = _mm_mul_ps(u, _mm_sub_ps(rb, ra));
    let w = exp2(v);
    _mm_div_ps(one, _mm_add_ps(one, w))
}

/// Transpose a 4x4 matrix of packed `f32` lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn transpose(v: [__m128; 4]) -> [__m128; 4] {
    // First interleave adjacent rows, then move the low/high halves into final
    // columns. This lets the mirrored BO1 block be written as contiguous rows.
    let lo = [_mm_unpacklo_ps(v[0], v[1]), _mm_unpacklo_ps(v[2], v[3])];
    let hi = [_mm_unpackhi_ps(v[0], v[1]), _mm_unpackhi_ps(v[2], v[3])];

    [
        _mm_movelh_ps(lo[0], lo[1]),
        _mm_movehl_ps(lo[1], lo[0]),
        _mm_movelh_ps(hi[0], hi[1]),
        _mm_movehl_ps(hi[1], hi[0]),
    ]
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
fn exp2(x: __m128) -> __m128 {
    // Clamp x so the reconstructed 2^n stays within normal f32 exponent bits.
    let x = _mm_min_ps(x, consts::INPUT_MAX);
    let x = _mm_max_ps(x, consts::INPUT_MIN);

    // Choose n = floor(x + 0.5), the nearest integer with ties toward +inf.
    let rounded = _mm_add_ps(x, consts::HALF);
    // Convert with truncation toward zero.
    let truncated = _mm_cvttps_epi32(rounded);
    let truncated = _mm_cvtepi32_ps(truncated);
    // For negative non-integers, floor(v) = trunc(v) - 1.
    let floor_correction = _mm_and_ps(_mm_cmpgt_ps(truncated, rounded), consts::ONE);
    let n = _mm_sub_ps(truncated, floor_correction);

    // Reduce to z = (x - n) * ln(2), so 2^x = 2^n * exp(z).
    let z = _mm_sub_ps(x, n);
    let z = _mm_mul_ps(z, consts::LN_2);

    // Approximate exp(z) as 1 + z + z^2 * P(z), with P evaluated in
    // independent pairs to shorten the multiplication dependency chain.
    let z2 = _mm_mul_ps(z, z);
    let p01 = _mm_add_ps(_mm_mul_ps(consts::POLY[0], z), consts::POLY[1]);
    let p23 = _mm_add_ps(_mm_mul_ps(consts::POLY[2], z), consts::POLY[3]);
    let p45 = _mm_add_ps(_mm_mul_ps(consts::POLY[4], z), consts::POLY[5]);
    let p0123 = _mm_add_ps(_mm_mul_ps(p01, z2), p23);
    let polynomial = _mm_add_ps(_mm_mul_ps(p0123, z2), p45);
    let y = _mm_add_ps(_mm_add_ps(_mm_mul_ps(polynomial, z2), z), consts::ONE);

    // Build 2^n directly as an IEEE-754 f32: exponent bits = n + bias, mantissa = 0.
    let exponent = _mm_cvttps_epi32(n);
    let biased_exponent = _mm_add_epi32(exponent, consts::EXPONENT_BIAS);
    let exponent_bits = _mm_slli_epi32(biased_exponent, exp2_constants::MANTISSA_BITS);
    let pow2n = _mm_castsi128_ps(exponent_bits);

    // 2^x = exp(z) * 2^n.
    _mm_mul_ps(y, pow2n)
}

mod consts {
    use std::mem::transmute;

    use super::*;

    const fn f32x4(data: [f32; 4]) -> __m128 {
        unsafe { transmute(data) }
    }

    const fn i32x4(data: [i32; 4]) -> __m128i {
        unsafe { transmute(data) }
    }

    pub const HALF: __m128 = f32x4([0.5; _]);
    pub const ONE: __m128 = f32x4([1.0; _]);
    pub const TWO: __m128 = f32x4([2.0; _]);
    pub const THREE: __m128 = f32x4([3.0; _]);
    pub const LN_2: __m128 = f32x4([std::f32::consts::LN_2; _]);
    pub const INPUT_MIN: __m128 = f32x4([exp2_constants::INPUT_MIN; _]);
    pub const INPUT_MAX: __m128 = f32x4([exp2_constants::INPUT_MAX; _]);
    pub const EXPONENT_BIAS: __m128i = i32x4([exp2_constants::EXPONENT_BIAS; _]);

    pub const POLY: [__m128; 6] = [
        f32x4([exp2_constants::POLY_0; _]),
        f32x4([exp2_constants::POLY_1; _]),
        f32x4([exp2_constants::POLY_2; _]),
        f32x4([exp2_constants::POLY_3; _]),
        f32x4([exp2_constants::POLY_4; _]),
        f32x4([exp2_constants::POLY_5; _]),
    ];
}
