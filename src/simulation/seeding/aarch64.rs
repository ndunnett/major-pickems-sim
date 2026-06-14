//! Optimised seeding implementation for aarch64.

#![allow(clippy::wildcard_imports)]

use std::arch::aarch64::*;

use crate::{datatypes::Set, simulation::sorting};

use super::{PackedSeeding, Seeding};

/// Return remaining team indices sorted by mid-stage seeding.
#[must_use]
#[inline]
pub fn seed_teams(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { neon_impl(remaining, diffs, opponents) }
    } else {
        super::scalar_impl(remaining, diffs, opponents)
    }
}

/// Return remaining team indices sorted by mid-stage seeding using NEON.
///
/// # Safety
///
/// Undefined behaviour on platforms that do not support NEON instructions.
/// Callers must gate this with `is_aarch64_feature_detected!("neon")` or an
/// equivalent guarantee.
#[target_feature(enable = "neon")]
#[must_use]
pub fn neon_impl(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    // Build the two sortable tiebreaker fields, then pack them with each team's
    // original seed so the SIMD sorting network can order all tiebreaks at once.
    let diffs = unsafe { vld1q_s8(diffs.as_ptr()) };
    let diff = diff(diffs);
    let buchholz = buchholz(diffs, opponents);
    let packed = pack(remaining, diff, buchholz);
    sorting::aarch64::sort_strip_neon(packed, remaining.len())
}

/// Convert raw win-loss differentials into sortable `u16` vector lanes.
///
/// The scalar representation stores lower values first, so each signed
/// differential is inverted around 15, then zero-extended into an unsigned lane
/// ready for packing.
#[inline]
#[target_feature(enable = "neon")]
fn diff(diffs: int8x16_t) -> [uint16x8_t; 2] {
    let sortable = vsubq_u8(consts::U15, vreinterpretq_u8_s8(diffs));
    [vmovl_u8(vget_low_u8(sortable)), vmovl_high_u8(sortable)]
}

/// Compute sortable Buchholz scores for all 16 teams.
///
/// Each opponent set is split into four 4-bit nibbles. For each nibble, a
/// 16-entry table contains the sum of the four corresponding differentials for
/// every possible opponent subset. The selected nibble sums are combined as
/// signed bytes, inverted around 15, and widened for packing.
#[inline]
#[target_feature(enable = "neon")]
fn buchholz(diffs: int8x16_t, opponents: &[Set; 16]) -> [uint16x8_t; 2] {
    // Preserve all opponent bitsets as two u16x8 vectors: teams 0-7 and teams 8-15.
    let bits = sets_to_vecs(opponents);

    // Build one lookup table for each 4-team block of possible opponents.
    let tables = [
        build_table::<0, 1, 2, 3>(diffs),
        build_table::<4, 5, 6, 7>(diffs),
        build_table::<8, 9, 10, 11>(diffs),
        build_table::<12, 13, 14, 15>(diffs),
    ];

    // Deinterleave each set's low and high bytes so every table lookup handles
    // all 16 teams at once.
    let low = vuzp1q_u8(vreinterpretq_u8_u16(bits[0]), vreinterpretq_u8_u16(bits[1]));
    let high = vuzp2q_u8(vreinterpretq_u8_u16(bits[0]), vreinterpretq_u8_u16(bits[1]));
    let indices = [
        vandq_u8(low, consts::U15),
        vshrq_n_u8::<4>(low),
        vandq_u8(high, consts::U15),
        vshrq_n_u8::<4>(high),
    ];

    // Look up and add all four signed-byte contributions before widening.
    let scores = vaddq_s8(
        vaddq_s8(
            vqtbl1q_s8(tables[0], indices[0]),
            vqtbl1q_s8(tables[1], indices[1]),
        ),
        vaddq_s8(
            vqtbl1q_s8(tables[2], indices[2]),
            vqtbl1q_s8(tables[3], indices[3]),
        ),
    );

    let sortable = vsubq_u8(consts::U15, vreinterpretq_u8_s8(scores));
    [vmovl_u8(vget_low_u8(sortable)), vmovl_high_u8(sortable)]
}

/// Load the raw bits from 16 [`Set`] values into two `u16x8` vectors.
#[inline]
#[target_feature(enable = "neon")]
fn sets_to_vecs(sets: &[Set; 16]) -> [uint16x8_t; 2] {
    unsafe {
        [
            vld1q_u16(sets.as_ptr().cast()),
            vld1q_u16(sets.as_ptr().add(8).cast()),
        ]
    }
}

/// Build a 16-entry lookup table for one 4-team differential block.
///
/// Table index bits indicate which of the four teams are opponents. The selected
/// lanes contain the sum of the corresponding differentials.
#[inline]
#[target_feature(enable = "neon")]
fn build_table<const A: i32, const B: i32, const C: i32, const D: i32>(
    diffs: int8x16_t,
) -> int8x16_t {
    vaddq_s8(
        vaddq_s8(
            vandq_s8(vdupq_laneq_s8::<A>(diffs), consts::NIBBLE[0]),
            vandq_s8(vdupq_laneq_s8::<B>(diffs), consts::NIBBLE[1]),
        ),
        vaddq_s8(
            vandq_s8(vdupq_laneq_s8::<C>(diffs), consts::NIBBLE[2]),
            vandq_s8(vdupq_laneq_s8::<D>(diffs), consts::NIBBLE[3]),
        ),
    )
}

/// Pack sort keys for all teams and mark eliminated teams as inactive.
///
/// Each active lane is encoded as `(diff << 10) | (buchholz << 5) | seed`.
/// Inactive lanes are set to `u16::MAX` so they sort after all active teams.
#[inline]
#[target_feature(enable = "neon")]
fn pack(remaining: Set, diff: [uint16x8_t; 2], buchholz: [uint16x8_t; 2]) -> PackedSeeding {
    // Expand the remaining-team bitset into one mask lane per team.
    let remaining_bits = vdupq_n_u16(remaining.to_bits());
    let remaining = [
        vtstq_u16(remaining_bits, consts::BITS[0]),
        vtstq_u16(remaining_bits, consts::BITS[1]),
    ];

    // Compose the packed seeding keys for teams 0-7 and 8-15.
    let vecs = [
        vorrq_u16(
            vorrq_u16(vshlq_n_u16(diff[0], 10), vshlq_n_u16(buchholz[0], 5)),
            consts::INDICES[0],
        ),
        vorrq_u16(
            vorrq_u16(vshlq_n_u16(diff[1], 10), vshlq_n_u16(buchholz[1], 5)),
            consts::INDICES[1],
        ),
    ];

    let mut packed = PackedSeeding::new();

    unsafe {
        // Blend active packed keys with `u16::MAX` sentinels for eliminated teams.
        vst1q_u16(
            packed.as_mut_ptr(),
            vbslq_u16(remaining[0], vecs[0], consts::MAX),
        );
        vst1q_u16(
            packed.as_mut_ptr().add(8),
            vbslq_u16(remaining[1], vecs[1], consts::MAX),
        );
    }

    packed
}

mod consts {
    use std::mem::transmute;

    use super::*;

    const fn u8x16(data: [u8; 16]) -> uint8x16_t {
        unsafe { transmute(data) }
    }

    const fn u16x8(data: [u16; 8]) -> uint16x8_t {
        unsafe { transmute(data) }
    }

    const fn u128_to_i8x16(data: u128) -> int8x16_t {
        unsafe { transmute(data) }
    }

    pub const MAX: uint16x8_t = u16x8([u16::MAX; _]);
    pub const U15: uint8x16_t = u8x16([15; _]);

    /// Team index lanes used as the final packed-key tiebreaker.
    pub const INDICES: [uint16x8_t; 2] = [
        u16x8([0, 1, 2, 3, 4, 5, 6, 7]),
        u16x8([8, 9, 10, 11, 12, 13, 14, 15]),
    ];

    /// One-hot team bit lanes used to expand a [`Set`] into a SIMD mask.
    ///
    /// Testing these lanes against a broadcast set bitfield reveals which teams
    /// are still active.
    pub const BITS: [uint16x8_t; 2] = [
        u16x8([1, 2, 4, 8, 16, 32, 64, 128]),
        u16x8([256, 512, 1024, 2048, 4096, 8192, 16384, 32768]),
    ];

    /// Byte masks for building 4-bit Buchholz lookup tables.
    ///
    /// Each vector corresponds to one bit of a nibble. Lanes whose table index
    /// contains that bit are `0xFF`; other lanes are `0x00`, allowing a broadcast
    /// differential to contribute only to matching table entries.
    pub const NIBBLE: [int8x16_t; 4] = [
        u128_to_i8x16(0xFF00_FF00_FF00_FF00_FF00_FF00_FF00_FF00),
        u128_to_i8x16(0xFFFF_0000_FFFF_0000_FFFF_0000_FFFF_0000),
        u128_to_i8x16(0xFFFF_FFFF_0000_0000_FFFF_FFFF_0000_0000),
        u128_to_i8x16(0xFFFF_FFFF_FFFF_FFFF_0000_0000_0000_0000),
    ];
}
