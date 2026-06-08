//! Optimised seeding implementation for x86.

#![allow(clippy::wildcard_imports)]

cfg_select! {
    target_arch = "x86" => {
        use std::arch::x86::*;
    }
    target_arch = "x86_64" => {
        use std::arch::x86_64::*;
    }
}

use crate::{datatypes::Set, simulation::sorting};

use super::{ByteMasks, PackedSeeding, Seeding};

/// Return remaining team indices sorted by mid-stage seeding.
#[cfg(target_arch = "x86")]
#[must_use]
#[inline]
pub fn seed_teams(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    if is_x86_feature_detected!("sse2") {
        unsafe { sse2_impl(remaining, diffs, opponents) }
    } else {
        super::scalar_impl(remaining, diffs, opponents)
    }
}

/// Return remaining team indices sorted by mid-stage seeding using SSE2.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support SSE2 instructions.
#[target_feature(enable = "sse2")]
#[must_use]
pub fn sse2_impl(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    let mut buchholz_scores = [0; 16];
    let diffs_vector = unsafe { _mm_loadu_si128(diffs.as_ptr().cast()) };

    for index in remaining {
        let i = index.to_usize();
        let opps = opponents[i];

        buchholz_scores[i] = unsafe {
            // Convert the 16 opponent bits into 16 byte masks: `0x00` for absent
            // and `0xFF` for present.
            let low_mask = _mm_loadl_epi64(ByteMasks::low_ptr(opps).cast());
            let high_mask = _mm_loadl_epi64(ByteMasks::high_ptr(opps).cast());
            let mask = _mm_unpacklo_epi64(low_mask, high_mask);

            // Zero out non-opponents while preserving selected signed bytes in
            // two's-complement form.
            let selected = _mm_and_si128(diffs_vector, mask);

            // `_mm_sad_epu8` sums unsigned byte values into two elements. Because the
            // total score fits in `i8`, the sum of the low bytes is the signed total.
            let halves = _mm_sad_epu8(selected, _mm_setzero_si128());
            let sum = _mm_add_epi64(halves, _mm_shuffle_epi32::<2>(halves));
            _mm_cvtsi128_si32(sum) as i8
        };
    }

    let mut packed = PackedSeeding::new();

    unsafe {
        let buchholz = _mm_loadu_si128(buchholz_scores.as_ptr().cast());
        let zero = _mm_setzero_si128();
        let fifteen = _mm_set1_epi16(15);

        // Sign-extend each i8 score into i16 lanes before inverting the sort order.
        let diff_sign = _mm_cmpgt_epi8(zero, diffs_vector);
        let diff_lo = _mm_sub_epi16(fifteen, _mm_unpacklo_epi8(diffs_vector, diff_sign));
        let diff_hi = _mm_sub_epi16(fifteen, _mm_unpackhi_epi8(diffs_vector, diff_sign));

        let buchholz_sign = _mm_cmpgt_epi8(zero, buchholz);
        let buchholz_lo = _mm_sub_epi16(fifteen, _mm_unpacklo_epi8(buchholz, buchholz_sign));
        let buchholz_hi = _mm_sub_epi16(fifteen, _mm_unpackhi_epi8(buchholz, buchholz_sign));

        let index_lo = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        let index_hi = _mm_set_epi16(15, 14, 13, 12, 11, 10, 9, 8);

        // Pack win-loss, Buchholz, and initial seed into the same u16 key used
        // by the scalar implementation.
        let packed_lo = _mm_or_si128(
            _mm_or_si128(_mm_slli_epi16(diff_lo, 10), _mm_slli_epi16(buchholz_lo, 5)),
            index_lo,
        );
        let packed_hi = _mm_or_si128(
            _mm_or_si128(_mm_slli_epi16(diff_hi, 10), _mm_slli_epi16(buchholz_hi, 5)),
            index_hi,
        );

        _mm_storeu_si128(packed.as_mut_ptr().cast(), packed_lo);
        _mm_storeu_si128(packed.as_mut_ptr().add(8).cast(), packed_hi);
    }

    // Non-remaining teams sort after all real packed seeds.
    for seed in remaining.inverted() {
        packed[seed.to_usize()] = u16::MAX;
    }

    sorting::x86::sort_strip_sse2(packed, remaining.len())
}
