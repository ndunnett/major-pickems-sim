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
    // Build the two sortable tiebreaker fields, then pack them with each team's
    // original seed so the SIMD sorting network can order all tiebreaks at once.
    let diffs = unsafe { _mm_loadu_si128(diffs.as_ptr().cast()) };
    let diff = diff(diffs);
    let buchholz = buchholz(diffs, opponents);
    let packed = pack(remaining, diff, buchholz);
    sorting::x86_64::sort_strip_avx2(packed, remaining.len())
}

/// Convert raw win-loss differentials into sortable `u16` vector lanes.
///
/// The scalar representation stores lower values first, so each signed
/// differential is inverted around 15, then zero-extended into an unsigned
/// lane ready for packing.
#[inline]
#[target_feature(enable = "avx2")]
fn diff(diffs: __m128i) -> __m256i {
    _mm256_cvtepu8_epi16(_mm_sub_epi8(consts::I15, diffs))
}

/// Compute sortable Buchholz scores for all 16 teams.
///
/// Each opponent set is split into four 4-bit nibbles. Pairs of 16-entry
/// tables contain the sums of the corresponding differentials for every
/// possible opponent subset. The selected nibble sums are combined, inverted
/// around 15, and widened into unsigned lanes ready for packing.
#[inline]
#[target_feature(enable = "avx2")]
fn buchholz(diffs: __m128i, opponents: &[Set; 16]) -> __m256i {
    // Duplicate the 16 differentials into both 128-bit lanes so two nibble
    // lookup tables can be built in parallel.
    let diffs = _mm256_broadcastsi128_si256(diffs);

    let tables = [
        build_tables(diffs, consts::TABLES[0]),
        build_tables(diffs, consts::TABLES[1]),
    ];

    let bits = unsafe { _mm256_loadu_si256(opponents.as_ptr().cast()) };

    // Extract the four opponent-set nibbles and pack them into byte indices.
    // The qword permutation groups indices with their corresponding tables.
    let indices = [
        _mm256_permute4x64_epi64::<{ consts::PERMUTE }>(_mm256_packus_epi16(
            _mm256_and_si256(bits, consts::U15),
            _mm256_and_si256(_mm256_srli_epi16::<4>(bits), consts::U15),
        )),
        _mm256_permute4x64_epi64::<{ consts::PERMUTE }>(_mm256_packus_epi16(
            _mm256_and_si256(_mm256_srli_epi16::<8>(bits), consts::U15),
            _mm256_srli_epi16::<12>(bits),
        )),
    ];

    // Look up two nibble contributions in each 128-bit lane and add the pairs.
    let scores = _mm256_add_epi8(
        _mm256_shuffle_epi8(tables[0], indices[0]),
        _mm256_shuffle_epi8(tables[1], indices[1]),
    );

    // Combine the two lane-local sums, invert the signed byte scores, and widen
    // the resulting non-negative values into the packed key's u16 lanes.
    _mm256_cvtepu8_epi16(_mm_sub_epi8(
        consts::I15,
        _mm_add_epi8(
            _mm256_castsi256_si128(scores),
            _mm256_extracti128_si256::<1>(scores),
        ),
    ))
}

/// Build two 16-entry signed-byte subset-sum tables in parallel.
///
/// Each shuffle mask selects a differential only in table entries whose index
/// contains the corresponding team bit. Adding the four shuffled vectors
/// produces every subset sum for two adjacent 4-team blocks.
#[inline]
#[target_feature(enable = "avx2")]
fn build_tables(diffs: __m256i, masks: [__m256i; 4]) -> __m256i {
    _mm256_add_epi8(
        _mm256_add_epi8(
            _mm256_shuffle_epi8(diffs, masks[0]),
            _mm256_shuffle_epi8(diffs, masks[1]),
        ),
        _mm256_add_epi8(
            _mm256_shuffle_epi8(diffs, masks[2]),
            _mm256_shuffle_epi8(diffs, masks[3]),
        ),
    )
}

/// Pack sort keys for all teams and mark eliminated teams as inactive.
///
/// Each active lane is encoded as `(diff << 10) | (buchholz << 5) | seed`.
/// Inactive lanes are set to `u16::MAX` so they sort after all active teams.
#[inline]
#[target_feature(enable = "avx2")]
fn pack(remaining: Set, diff: __m256i, buchholz: __m256i) -> PackedSeeding {
    let mut packed = PackedSeeding::new();

    unsafe {
        // Compose the packed seeding keys for all 16 teams.
        let unmasked = _mm256_or_si256(
            _mm256_or_si256(_mm256_slli_epi16(diff, 10), _mm256_slli_epi16(buchholz, 5)),
            consts::INDICES,
        );

        // Expand the remaining-team bitset into one mask lane per team.
        let low_mask = _mm_loadl_epi64(ByteMasks::low_ptr(remaining).cast());
        let high_mask = _mm_loadl_epi64(ByteMasks::high_ptr(remaining).cast());
        let mask = _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(low_mask, high_mask));

        // Blend active packed keys with `u16::MAX` sentinels for eliminated teams.
        let packed_vector = _mm256_or_si256(
            _mm256_and_si256(mask, unmasked),
            _mm256_andnot_si256(mask, consts::MAX),
        );

        _mm256_storeu_si256(packed.as_mut_ptr().cast(), packed_vector);
    }

    packed
}

mod consts {
    use std::mem::transmute;

    use super::*;

    const fn i8x16(data: [i8; 16]) -> __m128i {
        unsafe { transmute(data) }
    }

    const fn u16x16(data: [u16; 16]) -> __m256i {
        unsafe { transmute(data) }
    }

    const fn table_mask(lo_base: i8, hi_base: i8, bit: usize) -> __m256i {
        let mut mask = [-1; 32];
        let mut i = 0;

        while i < 16 {
            if i & (1 << bit) != 0 {
                mask[i] = lo_base + bit as i8;
                mask[i + 16] = hi_base + bit as i8;
            }
            i += 1;
        }

        unsafe { transmute(mask) }
    }

    const fn table_masks(lo_base: i8, hi_base: i8) -> [__m256i; 4] {
        [
            table_mask(lo_base, hi_base, 0),
            table_mask(lo_base, hi_base, 1),
            table_mask(lo_base, hi_base, 2),
            table_mask(lo_base, hi_base, 3),
        ]
    }

    pub const MAX: __m256i = u16x16([u16::MAX; _]);
    pub const U15: __m256i = u16x16([15; _]);
    pub const I15: __m128i = i8x16([15; _]);
    pub const PERMUTE: i32 = 0b1101_1000;
    pub const INDICES: __m256i = u16x16([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

    /// Shuffle masks for building pairs of 4-bit Buchholz lookup tables.
    ///
    /// The first group builds tables for teams 0-3 and 4-7; the second builds
    /// tables for teams 8-11 and 12-15. A negative shuffle index contributes
    /// zero to table entries that do not contain that team's bit.
    pub const TABLES: [[__m256i; 4]; 2] = [table_masks(0, 4), table_masks(8, 12)];
}
