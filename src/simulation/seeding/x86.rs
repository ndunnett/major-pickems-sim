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

use std::mem::transmute;

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
/// Undefined behaviour on platforms that do not support SSE2 instructions. Callers
/// must gate this with `is_x86_feature_detected!("sse2")` or an equivalent guarantee.
#[target_feature(enable = "sse2")]
#[must_use]
pub fn sse2_impl(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    // Build the two sortable tiebreaker fields, then pack them with each team's
    // original seed so the SIMD sorting network can order all tiebreaks at once.
    let diffs = unsafe { _mm_loadu_si128(diffs.as_ptr().cast()) };
    let widened_diffs = widen_diffs(diffs);
    let buchholz = buchholz(diffs, opponents);
    let packed = pack(remaining, widened_diffs, buchholz);
    sorting::x86::sort_strip_sse2(packed, remaining.len())
}

/// Convert raw win-loss differentials into sortable `u16` vector lanes.
#[inline]
#[target_feature(enable = "sse2")]
fn widen_diffs(diffs: __m128i) -> [__m128i; 2] {
    let sortable = _mm_sub_epi8(consts::I15, diffs);

    [
        _mm_unpacklo_epi8(sortable, _mm_setzero_si128()),
        _mm_unpackhi_epi8(sortable, _mm_setzero_si128()),
    ]
}

/// Compute sortable Buchholz scores for all 16 teams.
#[inline]
#[target_feature(enable = "sse2")]
fn buchholz(diffs: __m128i, opps: &[Set; 16]) -> [__m128i; 2] {
    let groups = [
        group(diffs, opps[0], opps[1], opps[2], opps[3]),
        group(diffs, opps[4], opps[5], opps[6], opps[7]),
        group(diffs, opps[8], opps[9], opps[10], opps[11]),
        group(diffs, opps[12], opps[13], opps[14], opps[15]),
    ];

    let scores = unsafe { _mm_loadu_si128(groups.as_ptr().cast()) };
    let sortable = _mm_sub_epi8(consts::I15, scores);

    [
        _mm_unpacklo_epi8(sortable, _mm_setzero_si128()),
        _mm_unpackhi_epi8(sortable, _mm_setzero_si128()),
    ]
}

/// Compute four Buchholz scores and return them in the low bytes.
#[inline]
#[target_feature(enable = "sse2")]
fn group(diffs: __m128i, a: Set, b: Set, c: Set, d: Set) -> [u16; 2] {
    let lo = _mm_unpacklo_epi8(sum(diffs, a), sum(diffs, b));
    let hi = _mm_unpacklo_epi8(sum(diffs, c), sum(diffs, d));
    unsafe { transmute(_mm_cvtsi128_si32(_mm_unpacklo_epi16(lo, hi))) }
}

/// Sum the signed differentials for one team's opponent set.
#[inline]
#[target_feature(enable = "sse2")]
fn sum(diffs: __m128i, opponents: Set) -> __m128i {
    let selected = _mm_and_si128(diffs, byte_mask(opponents));
    let halves = _mm_sad_epu8(selected, _mm_setzero_si128());
    _mm_add_epi64(halves, _mm_shuffle_epi32::<2>(halves))
}

/// Expand a [`Set`] into byte masks: `0x00` for absent and `0xFF` for present.
#[inline]
#[target_feature(enable = "sse2")]
fn byte_mask(set: Set) -> __m128i {
    unsafe {
        _mm_unpacklo_epi64(
            _mm_loadl_epi64(ByteMasks::low_ptr(set).cast()),
            _mm_loadl_epi64(ByteMasks::high_ptr(set).cast()),
        )
    }
}

/// Pack sort keys for all teams and mark eliminated teams as inactive.
///
/// Each active lane is encoded as `(diff << 10) | (buchholz << 5) | seed`.
/// Inactive lanes are set to `u16::MAX` so they sort after all active teams.
#[inline]
#[target_feature(enable = "sse2")]
fn pack(remaining: Set, diffs: [__m128i; 2], buchholz: [__m128i; 2]) -> PackedSeeding {
    let vecs = {
        // Compose the packed seeding keys for teams 0-7 and 8-15.
        let packed = [
            _mm_or_si128(
                _mm_or_si128(_mm_slli_epi16(diffs[0], 10), _mm_slli_epi16(buchholz[0], 5)),
                consts::INDICES[0],
            ),
            _mm_or_si128(
                _mm_or_si128(_mm_slli_epi16(diffs[1], 10), _mm_slli_epi16(buchholz[1], 5)),
                consts::INDICES[1],
            ),
        ];

        // Expand the remaining-team bitset into one mask lane per team.
        let mask = byte_mask(remaining);
        let masks = [_mm_unpacklo_epi8(mask, mask), _mm_unpackhi_epi8(mask, mask)];

        [
            // Blend active packed keys with `u16::MAX` sentinels for eliminated teams.
            _mm_or_si128(
                _mm_and_si128(masks[0], packed[0]),
                _mm_andnot_si128(masks[0], consts::MAX),
            ),
            _mm_or_si128(
                _mm_and_si128(masks[1], packed[1]),
                _mm_andnot_si128(masks[1], consts::MAX),
            ),
        ]
    };

    let mut packed = PackedSeeding::new();
    let ptr = packed.as_aligned_mut_ptr().cast::<__m128i>();

    unsafe {
        _mm_store_si128(ptr, vecs[0]);
        _mm_store_si128(ptr.add(1), vecs[1]);
    }

    packed
}

mod consts {
    use std::mem::transmute;

    use super::*;

    pub const MAX: __m128i = unsafe { transmute::<[u16; 8], _>([u16::MAX; _]) };
    pub const I15: __m128i = unsafe { transmute::<[i8; 16], _>([15; _]) };

    pub const INDICES: [__m128i; 2] = unsafe {
        transmute::<[u16; 16], _>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    };
}
