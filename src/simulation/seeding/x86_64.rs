//! Optimised seeding implementation for x86 and x86_64.

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

use std::mem::transmute;

use crate::datatypes::{Index, Set};

use super::{INITIAL_SEED_MASK, Seeding, sort};

/// Return remaining team indices sorted by mid-stage seeding.
#[must_use]
#[inline]
pub fn seed_teams(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    if is_x86_feature_detected!("avx2") && remaining.len() > 8 {
        unsafe { avx2_impl(remaining, diffs, opponents) }
    } else {
        unsafe { sse2_impl(remaining, diffs, opponents) }
    }
}

/// Return remaining team indices sorted by mid-stage seeding.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support SSE2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("sse2")` or an equivalent guarantee.
#[target_feature(enable = "sse2")]
#[must_use]
pub unsafe fn sse2_impl(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
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

    let mut packed = [0; 16];

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

    let len = remaining.len();
    sort(&mut packed, len);

    // Strip back down to just the zero-based initial seed.
    for packed_seed in &mut packed[..len] {
        *packed_seed &= INITIAL_SEED_MASK;
    }

    // `Index` is a transparent newtype of `u16`; the active prefix has been masked
    // down to only the initial seed, which is known to be in `0..16`.
    Seeding {
        len,
        data: unsafe { transmute::<[u16; 16], [Index; 16]>(packed) },
    }
}

/// Return remaining team indices sorted by mid-stage seeding.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support AVX2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("avx2")` or an equivalent guarantee.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn avx2_impl(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    let len = remaining.len();
    let diffs_vector = unsafe { _mm_loadu_si128(diffs.as_ptr().cast()) };
    let mut packed = [0; 16];

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

    sort(&mut packed, len);

    // Strip back down to just the zero-based initial seed.
    for packed_seed in &mut packed[..len] {
        *packed_seed &= INITIAL_SEED_MASK;
    }

    // `Index` is a transparent newtype of `u16`; the active prefix has been masked
    // down to only the initial seed, which is known to be in `0..16`.
    Seeding {
        len,
        data: unsafe { transmute::<[u16; 16], [Index; 16]>(packed) },
    }
}

/// Lookup table for expanding opponent bits into byte masks.
///
/// Each 8-bit index maps to eight bytes. A set bit becomes `0xFF`; an unset
/// bit becomes `0x00`.
#[repr(align(64))]
struct ByteMasks([i64; 256]);

impl ByteMasks {
    const MASKS: Self = {
        // Mask to copy each source bit into a separate byte lane when the
        // input byte is multiplied by it. The one-bit gaps prevent adjacent
        // source bits from carrying into each other.
        let spread_mask = {
            let mut mask = 1;
            let mut i = 0;

            while i < 7 {
                mask = (mask << 9) + 1;
                i += 1;
            }

            mask
        };

        // Mask to select the high bit from each byte lane after the spread multiply.
        let high_bits_mask = {
            let mut mask = 0x80;
            let mut i = 0;

            while i < 8 {
                mask = (mask << 8) + 0x80;
                i += 1;
            }

            mask
        };

        // Final byte mask array, each mask will represent half a set with each
        // bit in the set expanded to a byte.
        // ie. index `145` => half-set `0b1001_0001` => mask `0xFF00_00FF_0000_00FF`
        let mut masks = [0; 256];
        let mut i = 0;

        while i < 256 {
            masks[i] = {
                // Mask off the high bits for each lane, shift them down to the low bit.
                let high_bits = ((i as u64).wrapping_mul(spread_mask) & high_bits_mask) >> 7;

                // Multiply each bit by `0xFF`, reorder to put bit 0 in the lowest byte address.
                high_bits.wrapping_mul(0xFF).swap_bytes().cast_signed()
            };

            i += 1;
        }

        Self(masks)
    };

    /// Returns a pointer to the mask for the low half of the set.
    #[inline]
    const fn low_ptr(set: Set) -> *const i64 {
        // Safety: the index is always in `0x00..=0xFF`.
        unsafe { Self::MASKS.0.as_ptr().add((set.to_bits() & 0xFF) as usize) }
    }

    /// Returns a pointer to the mask for the high half of the set.
    #[inline]
    const fn high_ptr(set: Set) -> *const i64 {
        // Safety: the index is always in `0x00..=0xFF`.
        unsafe { Self::MASKS.0.as_ptr().add((set.to_bits() >> 8) as usize) }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_feature = "avx2")]
    use super::avx2_impl;
    use super::{super::scalar_impl, Index, Set, sse2_impl};

    const DIFF_FIXTURES: [[i8; 16]; 5] = [
        [0; 16],
        [-3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -3, 3],
        [3, -3, 2, -2, 1, -1, 0, 0, -1, 1, -2, 2, -3, 3, 0, -3],
        [1, 1, 1, 1, -1, -1, -1, -1, 2, 2, -2, -2, 3, -3, 0, 0],
        [3, 3, 2, 2, 2, 1, 1, 1, -1, -1, -1, -2, -2, -2, -3, -3],
    ];

    fn set_from_bits(bits: u16) -> Set {
        Index::iter_all()
            .filter(|index| bits & index.bit_select() != 0)
            .collect()
    }

    #[test]
    #[cfg(target_feature = "sse2")]
    fn sse2_matches_scalar() {
        for diffs in DIFF_FIXTURES {
            let opponents = std::array::from_fn(|i| {
                let rotate = u32::try_from(i).unwrap();
                set_from_bits(0b0001_0010_1010_0101_u16.rotate_left(rotate))
            });

            for bits in [
                0x0000, 0x0001, 0x8000, 0x00FF, 0xFF00, 0x5555, 0xAAAA, 0xFFFF,
            ] {
                let remaining = set_from_bits(bits);
                let scalar = scalar_impl(remaining, &diffs, &opponents);
                let sse2 = unsafe { sse2_impl(remaining, &diffs, &opponents) };
                assert_eq!(&*scalar, &*sse2);
            }
        }
    }

    #[test]
    #[cfg(target_feature = "avx2")]
    fn avx2_matches_scalar() {
        for diffs in DIFF_FIXTURES {
            let opponents = std::array::from_fn(|i| {
                let rotate = u32::try_from(i).unwrap();
                set_from_bits(0b0001_0010_1010_0101_u16.rotate_left(rotate))
            });

            for bits in [
                0x0000, 0x0001, 0x8000, 0x00FF, 0xFF00, 0x5555, 0xAAAA, 0xFFFF,
            ] {
                let remaining = set_from_bits(bits);
                let scalar = scalar_impl(remaining, &diffs, &opponents);
                let avx2 = unsafe { avx2_impl(remaining, &diffs, &opponents) };
                assert_eq!(&*scalar, &*avx2);
            }
        }
    }
}
