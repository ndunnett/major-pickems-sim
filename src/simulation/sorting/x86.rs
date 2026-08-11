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

use crate::{
    datatypes::Index,
    simulation::seeding::{PackedSeeding, Seeding},
};

/// Sort and strip a packed seeding array using SSE2.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support SSE2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("sse2")` or an equivalent guarantee.
#[target_feature(enable = "sse2")]
#[must_use]
pub fn sort_strip_sse2(seeding: PackedSeeding, len: usize) -> Seeding {
    let vectors = load(seeding);
    let sorted = sort(vectors);
    let packed = pack(sorted);
    let stripped = strip(packed);

    // `Index` is a transparent newtype of `u16`; only the first `len` entries
    // are read after `from_indices_unchecked`.
    let data = unsafe { transmute::<[__m128i; 2], [Index; 16]>(stripped) };
    unsafe { Seeding::from_indices_unchecked(len, data) }
}

/// Loads a [`PackedSeeding`] into four SSE2 vectors.
#[inline]
#[target_feature(enable = "sse2")]
fn load(mut seeding: PackedSeeding) -> [__m128i; 4] {
    const SIGN_BIT: __m128i = unsafe { transmute::<[u32; 4], _>([0x8000; _]) };

    // Safety: `seeding` is 32-byte aligned, so both 16-byte loads are aligned.
    let ptr = seeding.as_aligned_mut_ptr().cast::<__m128i>();
    let v = unsafe { [_mm_load_si128(ptr), _mm_load_si128(ptr.add(1))] };

    // Bias unsigned values into signed order so SSE2 min/max can compare them directly.
    [
        _mm_xor_si128(_mm_unpacklo_epi16(v[0], _mm_setzero_si128()), SIGN_BIT),
        _mm_xor_si128(_mm_unpackhi_epi16(v[0], _mm_setzero_si128()), SIGN_BIT),
        _mm_xor_si128(_mm_unpacklo_epi16(v[1], _mm_setzero_si128()), SIGN_BIT),
        _mm_xor_si128(_mm_unpackhi_epi16(v[1], _mm_setzero_si128()), SIGN_BIT),
    ]
}

/// Sorts 4 vectors of 4 u16 values stored in u32 lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn sort(mut v: [__m128i; 4]) -> [__m128i; 4] {
    // Compare and exchange element wise to place smaller values in `v[0..2]`
    // and larger values in `v[2..4]`.
    v = compare_exchange(v);

    // Pair each lane in `v[0..2]` with the adjacent lane in `v[2..4]`.
    v[2] = _mm_shuffle_epi32::<0xB1>(v[2]);
    v[3] = _mm_shuffle_epi32::<0xB1>(v[3]);
    v = compare_exchange(v);

    // Split alternating lane pairs across the vectors for the next comparisons.
    v = double_shuffle::<0x88, 0xDD>(v);
    v = compare_exchange(v);

    // Reverse each group of four lanes in `v[2..4]`.
    v[2] = _mm_shuffle_epi32::<0x1B>(v[2]);
    v[3] = _mm_shuffle_epi32::<0x1B>(v[3]);
    v = compare_exchange(v);

    // Group the halves of each logical eight-lane vector before comparing them.
    v = double_shuffle::<0x44, 0xEE>(v);
    v = compare_exchange(v);

    // Interleave lanes across vectors to start merging the ordered groups.
    v = double_shuffle::<0xD8, 0x8D>(v);
    v = compare_exchange(v);

    // Reverse the logical second eight-lane vector.
    let tmp = v[2];
    v[2] = _mm_shuffle_epi32::<0x1B>(v[3]);
    v[3] = _mm_shuffle_epi32::<0x1B>(tmp);
    v = compare_exchange(v);

    // Continue the merge until every lane has met its network partners.
    v = double_shuffle::<0xD8, 0x8D>(v);
    v = compare_exchange(v);
    v = double_shuffle::<0xD8, 0x8D>(v);
    v = compare_exchange(v);

    // Put lanes with the same final local rank into matching positions in the vectors.
    v = [
        _mm_unpacklo_epi32(v[0], v[1]),
        _mm_unpackhi_epi32(v[1], v[0]),
        _mm_unpacklo_epi32(v[2], v[3]),
        _mm_unpackhi_epi32(v[3], v[2]),
    ];

    // Do the last broad compare between low-ranked and high-ranked candidates.
    v = double_shuffle::<0x88, 0xDD>(v);
    v = compare_exchange(v);

    // Transpose even and odd lanes between logical vectors so the outputs are in final sorted order.
    v = [
        _mm_unpacklo_epi32(v[0], v[2]),
        _mm_unpackhi_epi32(v[0], v[2]),
        _mm_unpacklo_epi32(v[1], v[3]),
        _mm_unpackhi_epi32(v[1], v[3]),
    ];

    [
        _mm_unpacklo_epi64(v[0], v[1]),
        _mm_unpacklo_epi64(v[2], v[3]),
        _mm_unpackhi_epi64(v[0], v[1]),
        _mm_unpackhi_epi64(v[2], v[3]),
    ]
}

/// Compares each pair of elements between 4 vectors, placing smaller values in
/// `v[0..2]` and larger values in `v[2..4]`.
#[inline]
#[target_feature(enable = "sse2")]
fn compare_exchange(v: [__m128i; 4]) -> [__m128i; 4] {
    [
        _mm_min_epi16(v[0], v[2]),
        _mm_min_epi16(v[1], v[3]),
        _mm_max_epi16(v[0], v[2]),
        _mm_max_epi16(v[1], v[3]),
    ]
}

/// Shuffles 4 vectors of 4 u32 lanes into 4 vectors.
#[inline]
#[target_feature(enable = "sse2")]
fn double_shuffle<const MASK_0: i32, const MASK_1: i32>(v: [__m128i; 4]) -> [__m128i; 4] {
    let lo = shuffle::<MASK_0>(v);
    let hi = shuffle::<MASK_1>(v);
    [lo[0], lo[1], hi[0], hi[1]]
}

/// Shuffles 4 vectors of 4 u32 lanes into 2 vectors.
#[inline]
#[target_feature(enable = "sse2")]
fn shuffle<const MASK: i32>(v: [__m128i; 4]) -> [__m128i; 2] {
    // SSE2 only provides this shuffle for floating point vectors, but it leaves
    // the contents of each lane unchanged.
    unsafe {
        [
            transmute::<__m128, __m128i>(_mm_shuffle_ps::<MASK>(
                transmute::<__m128i, __m128>(v[0]),
                transmute::<__m128i, __m128>(v[2]),
            )),
            transmute::<__m128, __m128i>(_mm_shuffle_ps::<MASK>(
                transmute::<__m128i, __m128>(v[1]),
                transmute::<__m128i, __m128>(v[3]),
            )),
        ]
    }
}

/// Packs four vectors of 4 u16 values into two vectors of 8 u16 lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn pack(v: [__m128i; 4]) -> [__m128i; 2] {
    // Sign extend each biased u16 before packing it back out of its u32 lane.
    [
        _mm_packs_epi32(
            _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(v[0])),
            _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(v[1])),
        ),
        _mm_packs_epi32(
            _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(v[2])),
            _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(v[3])),
        ),
    ]
}

/// Strip active prefixes from the packed seed values.
#[inline]
#[target_feature(enable = "sse2")]
fn strip(v: [__m128i; 2]) -> [__m128i; 2] {
    const MASK: __m128i = unsafe { transmute::<[u16; 8], _>([0x1F; _]) };
    [_mm_and_si128(v[0], MASK), _mm_and_si128(v[1], MASK)]
}
