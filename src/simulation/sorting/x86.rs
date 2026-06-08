//! Optimised sorting implementation for x86.

#![allow(clippy::wildcard_imports)]

cfg_select! {
    target_arch = "x86" => {
        use std::arch::x86::*;
    }
    target_arch = "x86_64" => {
        use std::arch::x86_64::*;
    }
}

use std::mem::transmute;

use crate::simulation::seeding::{PackedSeeding, Seeding};

/// Sort and strip a packed seeding array using SSE2.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support SSE2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("sse2")` or an equivalent guarantee.
#[target_feature(enable = "sse2")]
#[must_use]
pub fn sort_strip_sse2(mut seeding: PackedSeeding, len: usize) -> Seeding {
    // Safety: `seeding` is 32-byte aligned, `ptr` and all increments are guaranteed to be aligned on 16-byte boundaries.
    let ptr = seeding.as_aligned_mut_ptr().cast::<__m128i>();
    let lo = unsafe { _mm_load_si128(ptr.cast_const()) };
    let hi = unsafe { _mm_load_si128(ptr.add(1).cast_const()) };
    let zero = _mm_setzero_si128();

    // Split and widen 2 vectors of 8 u16 lanes to 4 vectors of 4 u32 lanes; SSE2 compares 32-bit lanes directly.
    let lo_lo = _mm_unpacklo_epi16(lo, zero);
    let lo_hi = _mm_unpackhi_epi16(lo, zero);
    let hi_lo = _mm_unpacklo_epi16(hi, zero);
    let hi_hi = _mm_unpackhi_epi16(hi, zero);

    // Sort all 16 lanes as 4 vectors of 4.
    let (lo_lo, lo_hi, hi_lo, hi_hi) = sort_u32x4x4(lo_lo, lo_hi, hi_lo, hi_hi);

    // Keep the low two bytes from every sorted u32 lane and rejoin the four-lane groups.
    let lo = _mm_unpacklo_epi64(pack(lo_lo), pack(lo_hi));
    let hi = _mm_unpacklo_epi64(pack(hi_lo), pack(hi_hi));

    // Strip out the active prefixes, write the result back into the original aligned buffer
    // and convert to `Seeding`.
    let seed_mask = _mm_set1_epi16(0x1F);
    let lo = _mm_and_si128(lo, seed_mask);
    let hi = _mm_and_si128(hi, seed_mask);

    unsafe {
        _mm_store_si128(ptr, lo);
        _mm_store_si128(ptr.add(1), hi);
    }

    unsafe { seeding.into_seeding_unchecked(len) }
}

/// Sorts 4 vectors of 4 u32 lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn sort_u32x4x4(
    mut lo_lo: __m128i,
    mut lo_hi: __m128i,
    mut hi_lo: __m128i,
    mut hi_hi: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    // Compare and exchange element wise to place smaller values in `lo` and larger values in `hi`.
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Pair each lane in `lo` with the adjacent lane in `hi` for the next compare stage.
    hi_lo = _mm_shuffle_epi32::<0b1011_0001>(hi_lo);
    hi_hi = _mm_shuffle_epi32::<0b1011_0001>(hi_hi);
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Split alternating lane pairs across the two vectors so the next comparisons line up.
    let (mut tmp_lo, mut tmp_hi) = (lo_lo, lo_hi);
    (lo_lo, lo_hi) = shuffle_u32x4x4::<0b1000_1000>(lo_lo, lo_hi, hi_lo, hi_hi);
    (hi_lo, hi_hi) = shuffle_u32x4x4::<0b1101_1101>(tmp_lo, tmp_hi, hi_lo, hi_hi);
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Reverse each group of four lanes in `hi`, producing the next set of network partners.
    hi_lo = _mm_shuffle_epi32::<0b0001_1011>(hi_lo);
    hi_hi = _mm_shuffle_epi32::<0b0001_1011>(hi_hi);
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Group the low and high halves of each logical eight-lane vector before comparing them.
    (tmp_lo, tmp_hi) = (lo_lo, lo_hi);
    (lo_lo, lo_hi) = shuffle_u32x4x4::<0b0100_0100>(lo_lo, lo_hi, hi_lo, hi_hi);
    (hi_lo, hi_hi) = shuffle_u32x4x4::<0b1110_1110>(tmp_lo, tmp_hi, hi_lo, hi_hi);
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Interleave lanes across vectors; this starts merging the independently ordered groups.
    (tmp_lo, tmp_hi) = (lo_lo, lo_hi);
    (lo_lo, lo_hi) = shuffle_u32x4x4::<0b1101_1000>(lo_lo, lo_hi, hi_lo, hi_hi);
    (hi_lo, hi_hi) = shuffle_u32x4x4::<0b1000_1101>(tmp_lo, tmp_hi, hi_lo, hi_hi);
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Reverse the logical second eight-lane vector, swapping halves because SSE2 has no 256-bit lane.
    tmp_lo = hi_lo;
    hi_lo = _mm_shuffle_epi32::<0b0001_1011>(hi_hi);
    hi_hi = _mm_shuffle_epi32::<0b0001_1011>(tmp_lo);
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Continue the merge with the same cross-vector interleave pattern.
    (tmp_lo, tmp_hi) = (lo_lo, lo_hi);
    (lo_lo, lo_hi) = shuffle_u32x4x4::<0b1101_1000>(lo_lo, lo_hi, hi_lo, hi_hi);
    (hi_lo, hi_hi) = shuffle_u32x4x4::<0b1000_1101>(tmp_lo, tmp_hi, hi_lo, hi_hi);
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Repeat the merge step until every lane has met the required network partners.
    (tmp_lo, tmp_hi) = (lo_lo, lo_hi);
    (lo_lo, lo_hi) = shuffle_u32x4x4::<0b1101_1000>(lo_lo, lo_hi, hi_lo, hi_hi);
    (hi_lo, hi_hi) = shuffle_u32x4x4::<0b1000_1101>(tmp_lo, tmp_hi, hi_lo, hi_hi);
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Put lanes with the same final local rank into matching positions in both vectors.
    (lo_lo, lo_hi) = (
        _mm_unpacklo_epi32(lo_lo, lo_hi),
        _mm_unpackhi_epi32(lo_hi, lo_lo),
    );

    (hi_lo, hi_hi) = (
        _mm_unpacklo_epi32(hi_lo, hi_hi),
        _mm_unpackhi_epi32(hi_hi, hi_lo),
    );

    // Do the last broad compare between low-ranked and high-ranked candidates.
    (tmp_lo, tmp_hi) = (lo_lo, lo_hi);
    (lo_lo, lo_hi) = shuffle_u32x4x4::<0b1000_1000>(lo_lo, lo_hi, hi_lo, hi_hi);
    (hi_lo, hi_hi) = shuffle_u32x4x4::<0b1101_1101>(tmp_lo, tmp_hi, hi_lo, hi_hi);
    compare_exchange_u32x4x4(&mut lo_lo, &mut lo_hi, &mut hi_lo, &mut hi_hi);

    // Swap every odd lane between logical vectors so the outputs are in final sorted order.
    let odd_lanes = _mm_set_epi32(-1, 0, -1, 0);
    let b1_lo = _mm_shuffle_epi32::<0b1011_0001>(lo_lo);
    let b1_hi = _mm_shuffle_epi32::<0b1011_0001>(lo_hi);
    let b2_lo = _mm_shuffle_epi32::<0b1011_0001>(hi_lo);
    let b2_hi = _mm_shuffle_epi32::<0b1011_0001>(hi_hi);

    lo_lo = blend_u32x4x2(odd_lanes, b2_lo, lo_lo);
    lo_hi = blend_u32x4x2(odd_lanes, b2_hi, lo_hi);
    hi_lo = blend_u32x4x2(odd_lanes, hi_lo, b1_lo);
    hi_hi = blend_u32x4x2(odd_lanes, hi_hi, b1_hi);

    (lo_lo, lo_hi, hi_lo, hi_hi)
}

/// Compares each pair of elements between 4 vectors, placing smaller values in `lo` and larger values in `hi`.
#[inline]
#[target_feature(enable = "sse2")]
fn compare_exchange_u32x4x4(
    lo_lo: &mut __m128i,
    lo_hi: &mut __m128i,
    hi_lo: &mut __m128i,
    hi_hi: &mut __m128i,
) {
    compare_exchange_u32x4x2(lo_lo, hi_lo);
    compare_exchange_u32x4x2(lo_hi, hi_hi);
}

/// Compares each pair of elements between 2 vectors, placing smaller values in `lo` and larger values in `hi`.
#[inline]
#[target_feature(enable = "sse2")]
fn compare_exchange_u32x4x2(lo: &mut __m128i, hi: &mut __m128i) {
    // Input values started as u16, so signed i32 comparison is equivalent to unsigned here.
    let gt = _mm_cmpgt_epi32(*lo, *hi);
    let tmp = *lo;
    *lo = blend_u32x4x2(gt, *hi, *lo);
    *hi = blend_u32x4x2(gt, tmp, *hi);
}

/// Blends 2 vectors, selecting elements from `a` using `mask`, otherwise selecting from `b`.
#[inline]
#[target_feature(enable = "sse2")]
fn blend_u32x4x2(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
    // SSE2 has no blend instruction, so build one with mask-and/or operations.
    _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
}

/// Shuffles 4 vectors of 4 u32 lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn shuffle_u32x4x4<const MASK: i32>(
    a: __m128i,
    b: __m128i,
    c: __m128i,
    d: __m128i,
) -> (__m128i, __m128i) {
    (shuffle_u32x4x2::<MASK>(a, c), shuffle_u32x4x2::<MASK>(b, d))
}

/// Shuffles 2 vectors of 4 u32 lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn shuffle_u32x4x2<const MASK: i32>(a: __m128i, b: __m128i) -> __m128i {
    // SSE2 only has the desired 2 vector 4x32-bit shuffle instruction for floating point values.
    // Shuffle only changes composition of elements within the vector, bit contents of each
    // element remain unchanged.
    unsafe {
        transmute(_mm_shuffle_ps::<MASK>(
            transmute::<__m128i, __m128>(a),
            transmute::<__m128i, __m128>(b),
        ))
    }
}

#[inline]
#[target_feature(enable = "sse2")]
fn pack(v: __m128i) -> __m128i {
    // Pick bytes 0..1 and 4..5 from the low two u32 lanes.
    let low = _mm_shufflelo_epi16::<0b0000_1000>(v);
    // Shift the high two u32 lanes down, then apply the same compaction.
    let high = _mm_shufflelo_epi16::<0b0000_1000>(_mm_srli_si128::<8>(v));
    // Combine the two compacted pairs into four contiguous u16 values in the low 64 bits.
    _mm_unpacklo_epi32(low, high)
}
