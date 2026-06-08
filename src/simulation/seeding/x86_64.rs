//! Optimised seeding implementation for x86_64.

#![allow(clippy::wildcard_imports)]

use std::arch::x86_64::*;

use crate::{datatypes::Set, simulation::sorting};

use super::{ByteMasks, PackedSeeding, Seeding};

/// Return remaining team indices sorted by mid-stage seeding.
#[must_use]
#[inline]
pub fn seed_teams(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    if is_x86_feature_detected!("avx2") {
        unsafe { avx2_impl(remaining, diffs, opponents) }
    } else {
        // Safety: SSE2 is enabled on x86_64 architectures by default
        unsafe { super::x86::sse2_impl(remaining, diffs, opponents) }
    }
}

/// Return remaining team indices sorted by mid-stage seeding using AVX2.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support AVX2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("avx2")` or an equivalent guarantee.
#[target_feature(enable = "avx2")]
#[must_use]
pub fn avx2_impl(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    let len = remaining.len();
    let diffs_vector = unsafe { _mm_loadu_si128(diffs.as_ptr().cast()) };
    let mut packed = PackedSeeding::new();

    unsafe {
        let fifteen = _mm256_set1_epi16(15);
        let buchholz = {
            macro_rules! cast {
                ($($i:expr),+ $(,)*) => {
                    _mm256_set_epi16($(opponents[$i].to_bits().cast_signed()),+)
                };
            }

            let opponent_bits = cast!(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

            let mut scores = _mm256_setzero_si256();

            #[allow(clippy::needless_range_loop)]
            for i in 0..16 {
                let bit = _mm256_set1_epi16((1_u16 << i).cast_signed());
                let has_opponent = _mm256_cmpeq_epi16(_mm256_and_si256(opponent_bits, bit), bit);
                let diff = _mm256_set1_epi16(i16::from(diffs[i]));
                scores = _mm256_add_epi16(scores, _mm256_and_si256(has_opponent, diff));
            }

            scores
        };

        // Sign-extend each i8 score into i16 lanes before inverting the sort order.
        let diff = _mm256_sub_epi16(fifteen, _mm256_cvtepi8_epi16(diffs_vector));
        let buchholz = _mm256_sub_epi16(fifteen, buchholz);
        let index = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        // Pack win-loss, Buchholz, and initial seed into the same u16 key used
        // by the scalar implementation.
        let packed_vector = _mm256_or_si256(
            _mm256_or_si256(_mm256_slli_epi16(diff, 10), _mm256_slli_epi16(buchholz, 5)),
            index,
        );
        let packed_vector = if len < 16 {
            let remaining_low_mask = _mm_loadl_epi64(ByteMasks::low_ptr(remaining).cast());
            let remaining_high_mask = _mm_loadl_epi64(ByteMasks::high_ptr(remaining).cast());
            let remaining_mask =
                _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(remaining_low_mask, remaining_high_mask));

            _mm256_or_si256(
                _mm256_and_si256(remaining_mask, packed_vector),
                _mm256_andnot_si256(remaining_mask, _mm256_set1_epi16(-1)),
            )
        } else {
            packed_vector
        };

        _mm256_storeu_si256(packed.as_mut_ptr().cast(), packed_vector);
    }

    sorting::x86_64::sort_strip_avx2(packed, len)
}
