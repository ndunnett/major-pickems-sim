//! Optimised probability-matrix implementation for aarch64.

#![allow(clippy::wildcard_imports)]

use std::{arch::aarch64::*, f32::consts::LN_2};

use crate::datatypes::{Rating, Sigma};

use super::exp2_constants::*;

/// Precalculate BO1 and BO3 win-probability matrices.
///
/// The result is `[probabilities_bo1, probabilities_bo3]`, and each matrix is
/// indexed by `[team_a][team_b]`.
#[must_use]
#[inline]
pub fn calculate_probabilities(ratings: [Rating; 16], sigma: Sigma) -> [[[f32; 16]; 16]; 2] {
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { neon_impl(ratings, sigma) }
    } else {
        super::scalar_impl(ratings, sigma)
    }
}

/// Precalculate BO1 and BO3 win-probability matrices using NEON.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support NEON instructions.
/// Callers must gate this with `is_aarch64_feature_detected!("neon")` or an
/// equivalent guarantee.
#[target_feature(enable = "neon")]
#[must_use]
pub fn neon_impl(ratings: [Rating; 16], sigma: Sigma) -> [[[f32; 16]; 16]; 2] {
    let mut r = [f32::NAN; 16];

    for i in 0..16 {
        r[i] = ratings[i].to_f32();
    }

    // u = log2(10) / sigma
    // v = (Rb - Ra) * u
    // w = 2^v
    // P = 1 / (1 + w)
    let u = vdupq_n_f32(std::f32::consts::LOG2_10 / sigma.to_f32());
    let one = vdupq_n_f32(1.0);
    let mut probabilities_bo1 = [[f32::NAN; 16]; 16];

    for i in 0..16 {
        let ra = vdupq_n_f32(r[i]);

        for j in (0..16).step_by(4) {
            unsafe {
                // Load four potential opponents and compute row `i` of the BO1 matrix.
                let rb = vld1q_f32(r.as_ptr().add(j));
                let v = vmulq_f32(u, vsubq_f32(rb, ra));
                let w = exp2(v);
                let p = vdivq_f32(one, vaddq_f32(one, w));
                vst1q_f32(probabilities_bo1[i].as_mut_ptr().add(j), p);
            }
        }
    }

    // a = P * P
    // b = 1 - P
    // Q = 2 * a * b + a
    let two = vdupq_n_f32(2.0);
    let mut probabilities_bo3 = [[f32::NAN; 16]; 16];

    for i in 0..16 {
        for j in (0..16).step_by(4) {
            unsafe {
                // Convert each BO1 lane into the corresponding BO3 series probability.
                let p = vld1q_f32(probabilities_bo1[i].as_ptr().add(j));
                let a = vmulq_f32(p, p);
                let b = vsubq_f32(one, p);
                let q = vaddq_f32(vmulq_f32(two, vmulq_f32(a, b)), a);
                vst1q_f32(probabilities_bo3[i].as_mut_ptr().add(j), q);
            }
        }
    }

    [probabilities_bo1, probabilities_bo3]
}

/// Approximate `2^x` for four packed `f32` lanes using NEON.
///
/// The approximation splits each input into an integer power of two and a small
/// exponential remainder:
///
/// `2^x = 2^n * exp((x - n) * ln(2))`, where `n = floor(x + 0.5)`.
#[target_feature(enable = "neon")]
fn exp2(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);

    // Clamp x so the reconstructed 2^n stays within normal f32 exponent bits.
    let x = vminq_f32(x, vdupq_n_f32(INPUT_MAX));
    let x = vmaxq_f32(x, vdupq_n_f32(INPUT_MIN));

    // Choose n = floor(x + 0.5), the nearest integer with ties toward +inf.
    let n = vrndmq_f32(vaddq_f32(x, vdupq_n_f32(0.5)));

    // Reduce to z = (x - n) * ln(2), so 2^x = 2^n * exp(z).
    let z = vmulq_f32(vsubq_f32(x, n), vdupq_n_f32(LN_2));

    // Approximate exp(z) as 1 + z + z^2 * P(z), with P evaluated by Horner's rule.
    let mut y = vdupq_n_f32(POLY_0);
    y = vfmaq_f32(vdupq_n_f32(POLY_1), y, z);
    y = vfmaq_f32(vdupq_n_f32(POLY_2), y, z);
    y = vfmaq_f32(vdupq_n_f32(POLY_3), y, z);
    y = vfmaq_f32(vdupq_n_f32(POLY_4), y, z);
    y = vfmaq_f32(vdupq_n_f32(POLY_5), y, z);
    y = vmulq_f32(y, z);
    y = vfmaq_f32(z, y, z);
    y = vaddq_f32(y, one);

    // Build 2^n directly as an IEEE-754 f32: exponent bits = n + bias, mantissa = 0.
    let exponent = vcvtq_s32_f32(n);
    let biased_exponent = vaddq_s32(exponent, vdupq_n_s32(EXPONENT_BIAS));
    let exponent_bits = vshlq_n_s32(biased_exponent, MANTISSA_BITS);
    let pow2n = vreinterpretq_f32_s32(exponent_bits);

    // 2^x = exp(z) * 2^n.
    vmulq_f32(y, pow2n)
}
