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
    let diff = diff(diffs);
    let buchholz = buchholz(diffs, opponents);
    let packed = pack(remaining, diff, buchholz);
    sorting::aarch64::sort_strip_neon(packed, remaining.len())
}

/// Convert raw win-loss differentials into sortable `u16` vector lanes.
///
/// The scalar representation stores lower values first, so each signed
/// differential is sign-extended to `i16`, inverted around 15, and reinterpreted
/// as an unsigned lane ready for packing.
#[inline]
#[target_feature(enable = "neon")]
fn diff(diffs: &[i8; 16]) -> [uint16x8_t; 2] {
    // Load all 16 signed byte differentials, then process the low and high
    // halves independently because the packed key uses 16-bit lanes.
    let diffs = unsafe { vld1q_s8(diffs.as_ptr()) };

    [
        vreinterpretq_u16_s16(vsubq_s16(consts::I15, vmovl_s8(vget_low_s8(diffs)))),
        vreinterpretq_u16_s16(vsubq_s16(consts::I15, vmovl_s8(vget_high_s8(diffs)))),
    ]
}

/// Compute sortable Buchholz scores for all 16 teams.
///
/// Each opponent set is split into four 4-bit nibbles. For each nibble, a
/// 16-entry table contains the sum of the four corresponding differentials for
/// every possible opponent subset. The selected nibble sums are added and then
/// inverted around 15 to match the packed sort order.
#[inline]
#[target_feature(enable = "neon")]
fn buchholz(diffs: &[i8; 16], opponents: &[Set; 16]) -> [uint16x8_t; 2] {
    // Preserve all opponent bitsets as two u16x8 vectors: teams 0-7 and teams 8-15.
    let bits = sets_to_vecs(opponents);

    // Build one lookup table for each 4-team block of possible opponents.
    let tables = [
        build_table(&diffs[0..4]),
        build_table(&diffs[4..8]),
        build_table(&diffs[8..12]),
        build_table(&diffs[12..16]),
    ];

    // Extract each 4-bit opponent subset and use it to table-lookup that block's
    // contribution to the team's Buchholz score.
    let scores = [
        select(tables[0], vandq_u16(bits[0], consts::U15)),
        select(tables[1], vandq_u16(vshrq_n_u16::<4>(bits[0]), consts::U15)),
        select(tables[2], vandq_u16(vshrq_n_u16::<8>(bits[0]), consts::U15)),
        select(tables[3], vshrq_n_u16::<12>(bits[0])),
        select(tables[0], vandq_u16(bits[1], consts::U15)),
        select(tables[1], vandq_u16(vshrq_n_u16::<4>(bits[1]), consts::U15)),
        select(tables[2], vandq_u16(vshrq_n_u16::<8>(bits[1]), consts::U15)),
        select(tables[3], vshrq_n_u16::<12>(bits[1])),
    ];

    // Add the four nibble contributions for teams 0-7 and 8-15 separately.
    [
        vreinterpretq_u16_s16(combine(&scores[0..4])),
        vreinterpretq_u16_s16(combine(&scores[4..8])),
    ]
}

/// Load the raw bits from 16 [`Set`] values into two `u16x8` vectors.
#[inline]
#[target_feature(enable = "neon")]
fn sets_to_vecs(sets: &[Set; 16]) -> [uint16x8_t; 2] {
    unsafe {
        [
            // Each unaligned u64 load pulls four adjacent `Set` bitsets. Pair two
            // loads to form one eight-lane vector.
            vcombine_u16(
                vcreate_u16(sets.as_ptr().cast::<u64>().read_unaligned()),
                vcreate_u16(sets.as_ptr().add(4).cast::<u64>().read_unaligned()),
            ),
            vcombine_u16(
                vcreate_u16(sets.as_ptr().add(8).cast::<u64>().read_unaligned()),
                vcreate_u16(sets.as_ptr().add(12).cast::<u64>().read_unaligned()),
            ),
        ]
    }
}

/// Build a 16-entry lookup table for one 4-team differential block.
///
/// Table index bits indicate which of the four teams are opponents. The selected
/// lanes contain the sum of the corresponding differentials.
#[inline]
#[target_feature(enable = "neon")]
fn build_table(diffs: &[i8]) -> uint8x16_t {
    // Broadcast each differential across all table lanes, mask it into only the
    // lanes whose index contains that team's bit, then add the four contributions.
    vreinterpretq_u8_s8(vaddq_s8(
        vaddq_s8(
            vandq_s8(vdupq_n_s8(diffs[0]), consts::NIBBLE[0]),
            vandq_s8(vdupq_n_s8(diffs[1]), consts::NIBBLE[1]),
        ),
        vaddq_s8(
            vandq_s8(vdupq_n_s8(diffs[2]), consts::NIBBLE[2]),
            vandq_s8(vdupq_n_s8(diffs[3]), consts::NIBBLE[3]),
        ),
    ))
}

/// Select signed byte scores from a nibble lookup table.
///
/// The `index` vector holds eight 4-bit table indices. The selected bytes are
/// sign-extended into `i16` lanes so later additions keep signed score semantics.
#[inline]
#[target_feature(enable = "neon")]
fn select(table: uint8x16_t, index: uint16x8_t) -> int16x8_t {
    // `vqtbl1_u8` operates on eight byte indices, so narrow the u16 indexes
    // before table lookup and sign-extend the resulting bytes afterward.
    vmovl_s8(vreinterpret_s8_u8(vqtbl1_u8(table, vmovn_u16(index))))
}

/// Combine four nibble-level Buchholz contributions into sortable lanes.
///
/// The raw signed sum is subtracted from 15 so ascending `u16` sort order places
/// stronger Buchholz scores before weaker ones.
#[inline]
#[target_feature(enable = "neon")]
fn combine(scores: &[int16x8_t]) -> int16x8_t {
    // Sum the four independent 4-team blocks for each lane, then invert the
    // result to match the packed key's ascending sort convention.
    vsubq_s16(
        consts::I15,
        vaddq_s16(
            vaddq_s16(scores[0], scores[1]),
            vaddq_s16(scores[2], scores[3]),
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
        vceqq_u16(vandq_u16(remaining_bits, consts::BITS[0]), consts::BITS[0]),
        vceqq_u16(vandq_u16(remaining_bits, consts::BITS[1]), consts::BITS[1]),
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

    const fn i16x8(data: [i16; 8]) -> int16x8_t {
        unsafe { transmute(data) }
    }

    const fn u16x8(data: [u16; 8]) -> uint16x8_t {
        unsafe { transmute(data) }
    }

    const fn u128_to_i8x16(data: u128) -> int8x16_t {
        unsafe { transmute(data) }
    }

    pub const MAX: uint16x8_t = u16x8([u16::MAX; _]);
    pub const U15: uint16x8_t = u16x8([15; _]);
    pub const I15: int16x8_t = i16x8([15; _]);

    /// Team index lanes used as the final packed-key tiebreaker.
    pub const INDICES: [uint16x8_t; 2] = [
        u16x8([0, 1, 2, 3, 4, 5, 6, 7]),
        u16x8([8, 9, 10, 11, 12, 13, 14, 15]),
    ];

    /// One-hot team bit lanes used to expand a [`Set`] into a SIMD mask.
    ///
    /// ANDing these lanes with a broadcast set bitfield reveals which teams are
    /// still active.
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
