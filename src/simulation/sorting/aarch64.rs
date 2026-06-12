//! Optimised sorting implementation for aarch64.

#![allow(clippy::wildcard_imports)]

use std::{arch::aarch64::*, mem::transmute};

use crate::{
    datatypes::Index,
    simulation::seeding::{PackedSeeding, Seeding},
};

/// Sort and strip a packed seeding array using NEON.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support NEON instructions.
/// Callers must gate this with `is_aarch64_feature_detected!("neon")` or an
/// equivalent guarantee.
#[target_feature(enable = "neon")]
#[must_use]
pub fn sort_strip_neon(seeding: PackedSeeding, len: usize) -> Seeding {
    let vectors = load(seeding);
    let sorted = sort(vectors);
    let stripped = strip(sorted);

    // `Index` is a transparent newtype of `u16`; only the first `len` entries
    // are read after `from_indices_unchecked`.
    let data = unsafe { transmute::<[uint16x8_t; 2], [Index; 16]>(stripped) };
    unsafe { Seeding::from_indices_unchecked(len, data) }
}

/// Loads a [`PackedSeeding`] into two NEON vectors.
#[inline]
#[target_feature(enable = "neon")]
fn load(mut seeding: PackedSeeding) -> [uint16x8_t; 2] {
    // Safety: `seeding` is 32-byte aligned, so both 16-byte loads are aligned.
    let ptr = seeding.as_aligned_mut_ptr().cast::<u16>();
    unsafe {
        [
            vld1q_u16(ptr.cast_const()),
            vld1q_u16(ptr.add(8).cast_const()),
        ]
    }
}

/// Sorts 2 vectors of 8 u16 lanes.
#[inline]
#[target_feature(enable = "neon")]
fn sort(mut v: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
    // Compare and exchange element wise to place smaller values in `v[0]` and larger values in `v[1]`.
    compare_exchange(&mut v);

    // Pair each lane in `v[0]` with the adjacent lane in `v[1]` for the next compare stage.
    v[1] = vrev32q_u16(v[1]);
    compare_exchange(&mut v);

    // Split alternating lane pairs across the two vectors so the next comparisons line up.
    double_shuffle(&mut v, consts::SHUFFLE_88, consts::SHUFFLE_DD);
    compare_exchange(&mut v);

    // Reverse each group of four lanes in `v[1]`, producing the next set of network partners.
    v[1] = vrev64q_u16(v[1]);
    compare_exchange(&mut v);

    // Group the low and high halves of each 128-bit lane before comparing them.
    double_shuffle(&mut v, consts::SHUFFLE_44, consts::SHUFFLE_EE);
    compare_exchange(&mut v);

    // Interleave lanes across vectors; this starts merging the independently ordered groups.
    double_shuffle(&mut v, consts::SHUFFLE_D8, consts::SHUFFLE_8D);
    compare_exchange(&mut v);

    // Reverse the whole second vector so low values in `v[0]` compare against high values in `v[1]`.
    v[1] = vrev64q_u16(vextq_u16::<4>(v[1], v[1]));
    compare_exchange(&mut v);

    // Continue the merge with the same cross-vector interleave pattern until every lane has met the required network partners.
    double_shuffle(&mut v, consts::SHUFFLE_D8, consts::SHUFFLE_8D);
    compare_exchange(&mut v);
    double_shuffle(&mut v, consts::SHUFFLE_D8, consts::SHUFFLE_8D);
    compare_exchange(&mut v);

    // Put lanes with the same final local rank into matching positions in both vectors.
    v = [
        vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(v[0]), consts::PERMUTE)),
        vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(v[1]), consts::PERMUTE)),
    ];

    // Do the last broad compare between low-ranked and high-ranked candidates.
    double_shuffle(&mut v, consts::SHUFFLE_88, consts::SHUFFLE_DD);
    compare_exchange(&mut v);

    // Transpose even and odd lanes between vectors so the outputs are in final sorted order.
    uint16x8x2_t(v[0], v[1]) = vtrnq_u16(v[0], v[1]);

    v
}

/// Compares each pair of elements between 2 vectors, places smaller values in `v[0]` and larger values in `v[1]`.
#[inline]
#[target_feature(enable = "neon")]
fn compare_exchange(v: &mut [uint16x8_t; 2]) {
    let tmp = v[0];
    v[0] = vminq_u16(v[0], v[1]);
    v[1] = vmaxq_u16(tmp, v[1]);
}

/// Shuffles 2 vectors of 8 u16 lanes into 2 vectors.
#[inline]
#[target_feature(enable = "neon")]
fn double_shuffle(v: &mut [uint16x8_t; 2], indices1: uint8x16_t, indices2: uint8x16_t) {
    let tmp = v[0];
    v[0] = shuffle(v[0], v[1], indices1);
    v[1] = shuffle(tmp, v[1], indices2);
}

/// Shuffles 2 vectors of 8 u16 lanes into 1 vector.
#[inline]
#[target_feature(enable = "neon")]
fn shuffle(lo: uint16x8_t, hi: uint16x8_t, indices: uint8x16_t) -> uint16x8_t {
    let table = uint8x16x2_t(vreinterpretq_u8_u16(lo), vreinterpretq_u8_u16(hi));
    vreinterpretq_u16_u8(vqtbl2q_u8(table, indices))
}

/// Strip active prefixes from the packed seed values.
#[inline]
#[target_feature(enable = "neon")]
fn strip(v: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
    [
        vandq_u16(v[0], consts::STRIP_MASK),
        vandq_u16(v[1], consts::STRIP_MASK),
    ]
}

mod consts {
    use super::*;

    /// Convert lane indices into byte indices suitable for NEON table lookups.
    const fn indices(lanes: [u8; 8]) -> uint8x16_t {
        let bytes = [
            lanes[0] * 2,
            lanes[0] * 2 + 1,
            lanes[1] * 2,
            lanes[1] * 2 + 1,
            lanes[2] * 2,
            lanes[2] * 2 + 1,
            lanes[3] * 2,
            lanes[3] * 2 + 1,
            lanes[4] * 2,
            lanes[4] * 2 + 1,
            lanes[5] * 2,
            lanes[5] * 2 + 1,
            lanes[6] * 2,
            lanes[6] * 2 + 1,
            lanes[7] * 2,
            lanes[7] * 2 + 1,
        ];

        unsafe { transmute(bytes) }
    }

    pub const SHUFFLE_88: uint8x16_t = indices([0, 2, 8, 10, 4, 6, 12, 14]);
    pub const SHUFFLE_DD: uint8x16_t = indices([1, 3, 9, 11, 5, 7, 13, 15]);
    pub const SHUFFLE_44: uint8x16_t = indices([0, 1, 8, 9, 4, 5, 12, 13]);
    pub const SHUFFLE_EE: uint8x16_t = indices([2, 3, 10, 11, 6, 7, 14, 15]);
    pub const SHUFFLE_D8: uint8x16_t = indices([0, 2, 9, 11, 4, 6, 13, 15]);
    pub const SHUFFLE_8D: uint8x16_t = indices([1, 3, 8, 10, 5, 7, 12, 14]);
    pub const PERMUTE: uint8x16_t = indices([0, 4, 1, 5, 6, 2, 7, 3]);
    pub const STRIP_MASK: uint16x8_t = unsafe { transmute([0x1F_u16; 8]) };
}
