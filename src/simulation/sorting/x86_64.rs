//! Optimised sorting implementation for x86_64.

#![allow(clippy::wildcard_imports)]

use std::{arch::x86_64::*, mem::transmute};

use crate::simulation::seeding::{PackedSeeding, Seeding};

/// Sort and strip a packed seeding array using AVX2.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support AVX2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("avx2")` or an equivalent guarantee.
#[target_feature(enable = "avx2")]
#[must_use]
pub fn sort_strip_avx2(mut seeding: PackedSeeding, len: usize) -> Seeding {
    // Safety: `seeding` is 32-byte aligned, `ptr` is guaranteed to be aligned on a 32-byte boundary.
    let ptr = seeding.as_aligned_mut_ptr().cast::<__m256i>();
    let packed = unsafe { _mm256_load_si256(ptr.cast_const()) };

    // Split and widen 1 vector of 16 u16 lanes to 2 vectors of 8 u32 lanes; AVX2 compares 32-bit lanes directly.
    let lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(packed));
    let hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(packed));

    // Sort all 16 lanes as 2 vectors of 8.
    let (lo, hi) = sort_u32x8x2(lo, hi);

    // Keep the low two bytes of each u32 lane and discard the widening padding bytes.
    let mask = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13, -1, -1,
        -1, -1, -1, -1, -1, -1,
    );

    // Pack the sorted u32 lanes back into one 16-lane u16 vector.
    let packed = _mm256_permute2x128_si256::<0x20>(
        // After the byte shuffle, each 128-bit half contains four compacted u16 values.
        // Reorder the 64-bit chunks so the first eight sorted values become contiguous.
        _mm256_permute4x64_epi64::<0b1101_1000>(_mm256_shuffle_epi8(lo, mask)),
        _mm256_permute4x64_epi64::<0b1101_1000>(_mm256_shuffle_epi8(hi, mask)),
    );

    // Strip out the active prefixes, write the result back into the original aligned buffer
    // and convert to `Seeding`.
    let stripped = _mm256_and_si256(packed, _mm256_set1_epi16(0x1F));
    unsafe { _mm256_store_si256(ptr, stripped) };
    unsafe { seeding.into_seeding_unchecked(len) }
}

/// Sorts 2 vectors of 8 u32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn sort_u32x8x2(mut lo: __m256i, mut hi: __m256i) -> (__m256i, __m256i) {
    // Compare and exchange element wise to place smaller values in `lo` and larger values in `hi`.
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Pair each lane in `lo` with the adjacent lane in `hi` for the next compare stage.
    hi = _mm256_shuffle_epi32::<0b1011_0001>(hi);
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Split alternating lane pairs across the two vectors so the next comparisons line up.
    let mut tmp = lo;
    lo = shuffle_u32x8x2::<0b1000_1000>(lo, hi);
    hi = shuffle_u32x8x2::<0b1101_1101>(tmp, hi);
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Reverse each group of four lanes in `hi`, producing the next set of network partners.
    hi = _mm256_shuffle_epi32::<0b0001_1011>(hi);
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Group the low and high halves of each 128-bit lane before comparing them.
    tmp = lo;
    lo = shuffle_u32x8x2::<0b0100_0100>(lo, hi);
    hi = shuffle_u32x8x2::<0b1110_1110>(tmp, hi);
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Interleave lanes across vectors; this starts merging the independently ordered groups.
    tmp = lo;
    lo = shuffle_u32x8x2::<0b1101_1000>(lo, hi);
    hi = shuffle_u32x8x2::<0b1000_1101>(tmp, hi);
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Reverse the whole second vector so low values in `lo` compare against high values in `hi`.
    hi = _mm256_permutevar8x32_epi32(hi, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Continue the merge with the same cross-vector interleave pattern.
    tmp = lo;
    lo = shuffle_u32x8x2::<0b1101_1000>(lo, hi);
    hi = shuffle_u32x8x2::<0b1000_1101>(tmp, hi);
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Repeat the merge step until every lane has met the required network partners.
    tmp = lo;
    lo = shuffle_u32x8x2::<0b1101_1000>(lo, hi);
    hi = shuffle_u32x8x2::<0b1000_1101>(tmp, hi);
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Put lanes with the same final local rank into matching positions in both vectors.
    lo = _mm256_permutevar8x32_epi32(lo, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));
    hi = _mm256_permutevar8x32_epi32(hi, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));

    // Do the last broad compare between low-ranked and high-ranked candidates.
    tmp = lo;
    lo = shuffle_u32x8x2::<0b1000_1000>(lo, hi);
    hi = shuffle_u32x8x2::<0b1101_1101>(tmp, hi);
    compare_exchange_u32x8x2(&mut lo, &mut hi);

    // Swap every odd lane between vectors so the two outputs are in final sorted order.
    let b1 = _mm256_shuffle_epi32::<0b1011_0001>(lo);
    let b2 = _mm256_shuffle_epi32::<0b1011_0001>(hi);
    lo = _mm256_blend_epi32::<0b1010_1010>(lo, b2);
    hi = _mm256_blend_epi32::<0b1010_1010>(b1, hi);

    (lo, hi)
}

/// Compares each pair of elements between 2 vectors, places smaller values in `lo` and larger values in `hi`.
#[inline]
#[target_feature(enable = "avx2")]
fn compare_exchange_u32x8x2(lo: &mut __m256i, hi: &mut __m256i) {
    let tmp = *lo;
    *lo = _mm256_min_epu32(*lo, *hi);
    *hi = _mm256_max_epu32(tmp, *hi);
}

/// Shuffles 2 vectors of 8 u32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn shuffle_u32x8x2<const MASK: i32>(a: __m256i, b: __m256i) -> __m256i {
    // AVX2 only has the desired 2 vector 8x32-bit shuffle instruction for floating point values.
    // Shuffle only changes composition of elements within the vector, bit contents of each
    // element remain unchanged.
    unsafe {
        transmute(_mm256_shuffle_ps::<MASK>(
            transmute::<__m256i, __m256>(a),
            transmute::<__m256i, __m256>(b),
        ))
    }
}
