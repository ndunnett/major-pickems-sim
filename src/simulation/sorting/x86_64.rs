//! Optimised sorting implementation for x86_64.

#![allow(clippy::wildcard_imports)]

use std::{arch::x86_64::*, mem::transmute};

use crate::{
    datatypes::Index,
    simulation::seeding::{PackedSeeding, Seeding},
};

/// Sort and strip a packed seeding array using AVX2.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support AVX2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("avx2")` or an equivalent guarantee.
#[target_feature(enable = "avx2")]
#[must_use]
pub fn sort_strip_avx2(seeding: PackedSeeding, len: usize) -> Seeding {
    let vectors = load(seeding);
    let sorted = sort(vectors);
    let packed = pack(sorted);
    let stripped = strip(packed);

    // `Index` is a transparent newtype of `u16`; only the first `len` entries
    // are read after `from_indices_unchecked`.
    let data = unsafe { transmute::<__m256i, [Index; 16]>(stripped) };
    unsafe { Seeding::from_indices_unchecked(len, data) }
}

/// Loads a [`PackedSeeding`] into two AVX2 vectors.
#[inline]
#[target_feature(enable = "avx2")]
fn load(mut seeding: PackedSeeding) -> [__m256i; 2] {
    // Safety: `seeding` is 32-byte aligned, so both 16-byte loads are aligned.
    let ptr = seeding.as_aligned_mut_ptr().cast::<__m128i>();

    unsafe {
        [
            _mm256_cvtepu16_epi32(_mm_load_si128(ptr)),
            _mm256_cvtepu16_epi32(_mm_load_si128(ptr.add(1))),
        ]
    }
}

/// Sorts 2 vectors of 8 u32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn sort(mut v: [__m256i; 2]) -> [__m256i; 2] {
    const REVERSE: __m256i = unsafe { transmute::<[u32; 8], _>([7, 6, 5, 4, 3, 2, 1, 0]) };

    // Compare and exchange element wise to place smaller values in `v[0]`
    // and larger values in `v[1]`.
    v = compare_exchange(v);

    // Pair each lane in `v[0]` with the adjacent lane in `v[1]`.
    v[1] = _mm256_shuffle_epi32::<0xB1>(v[1]);
    v = compare_exchange(v);

    // Split alternating lane pairs across the vectors for the next comparisons.
    v = double_shuffle::<0x88, 0xDD>(v);
    v = compare_exchange(v);

    // Reverse each group of four lanes in `v[1]`.
    v[1] = _mm256_shuffle_epi32::<0x1B>(v[1]);
    v = compare_exchange(v);

    // Group the low and high halves of each 128-bit lane before comparing them.
    v = double_shuffle::<0x44, 0xEE>(v);
    v = compare_exchange(v);

    // Interleave lanes across vectors to start merging the ordered groups.
    v = double_shuffle::<0xD8, 0x8D>(v);
    v = compare_exchange(v);

    // Reverse `v[1]` so low values compare against high values.
    v[1] = _mm256_permutevar8x32_epi32(v[1], REVERSE);
    v = compare_exchange(v);

    // Continue the merge until every lane has met its network partners.
    v = double_shuffle::<0xD8, 0x8D>(v);
    v = compare_exchange(v);
    v = double_shuffle::<0xD8, 0x8D>(v);
    v = compare_exchange(v);

    // Put lanes with the same final local rank into matching positions in both vectors.
    v = permute(v);

    // Do the last broad compare between low-ranked and high-ranked candidates.
    v = double_shuffle::<0x88, 0xDD>(v);
    v = compare_exchange(v);

    v
}

/// Compares each pair of elements between 2 vectors, placing smaller values in
/// `v[0]` and larger values in `v[1]`.
#[inline]
#[target_feature(enable = "avx2")]
fn compare_exchange(v: [__m256i; 2]) -> [__m256i; 2] {
    [_mm256_min_epu32(v[0], v[1]), _mm256_max_epu32(v[0], v[1])]
}

/// Shuffles 2 vectors of 8 u32 lanes into 2 vectors.
#[inline]
#[target_feature(enable = "avx2")]
fn double_shuffle<const MASK_0: i32, const MASK_1: i32>(v: [__m256i; 2]) -> [__m256i; 2] {
    [shuffle::<MASK_0>(v), shuffle::<MASK_1>(v)]
}

/// Shuffles 2 vectors of 8 u32 lanes into 1 vector.
#[inline]
#[target_feature(enable = "avx2")]
fn shuffle<const MASK: i32>(v: [__m256i; 2]) -> __m256i {
    // AVX2 only provides this shuffle for floating point vectors, but it leaves
    // the contents of each lane unchanged.
    unsafe {
        transmute(_mm256_shuffle_ps::<MASK>(
            transmute::<__m256i, __m256>(v[0]),
            transmute::<__m256i, __m256>(v[1]),
        ))
    }
}

/// Reorders each vector so lanes with the same final local rank line up before
/// the last broad compare stage.
#[inline]
#[target_feature(enable = "avx2")]
fn permute(v: [__m256i; 2]) -> [__m256i; 2] {
    const PERMUTE: __m256i = unsafe { transmute::<[u32; 8], _>([0, 4, 2, 6, 1, 5, 3, 7]) };

    [
        _mm256_permutevar8x32_epi32(v[0], PERMUTE),
        _mm256_permutevar8x32_epi32(v[1], PERMUTE),
    ]
}

/// Packs and transposes two vectors of 8 u32 lanes into one vector of 16 u16 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn pack(v: [__m256i; 2]) -> __m256i {
    const U16_MAX: __m256i = unsafe { transmute::<[u32; 8], _>([u16::MAX as u32; _]) };
    const PACK: __m256i = unsafe {
        transmute::<[i8; 32], _>([
            0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15, 0, 1, 8, 9, 4, 5, 12, 13, 2, 3,
            10, 11, 6, 7, 14, 15,
        ])
    };

    // `_mm256_packus_epi32` packs within each 128-bit half, so the byte shuffle
    // fixes the lane-local order into one contiguous sorted `u16` sequence.
    _mm256_shuffle_epi8(
        _mm256_packus_epi32(
            _mm256_and_si256(v[0], U16_MAX),
            _mm256_and_si256(v[1], U16_MAX),
        ),
        PACK,
    )
}

/// Strip active prefixes from the packed seed values.
#[inline]
#[target_feature(enable = "avx2")]
fn strip(v: __m256i) -> __m256i {
    const MASK: __m256i = unsafe { transmute::<[u16; 16], _>([0x1F; _]) };
    _mm256_and_si256(v, MASK)
}
