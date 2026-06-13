//! Return remaining team indices sorted by mid-stage seeding.
//!
//! 1. Current win-loss record
//! 2. Buchholz difficulty score (sum of win-loss record for each opponent faced)
//! 3. Initial seeding
//!
//! [Rules and Regs - Mid-stage Seed Calculation](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#Mid-Stage-Seed-Calculation)
//!
//! Each piece of seeding information is small enough to fit into 5 bits.
//! Bit-pack each piece into a 16-bit unsigned integer so that one sort
//! applies every tiebreak in priority order:
//!
//! ```text
//! [15] [14 13 12 11 10] [9 8 7 6 5] [4 3 2 1 0]
//!  --   --------------   ---------   ---------
//!   |          |             |           |
//! Spare        |       2. Buchholz       |
//!         1. Win-loss             3. Initial seed
//! ```
//!
//! After sorting, the lowest 5 bits are masked out to extract the initial
//! seed which identifies the team.

#![allow(clippy::cast_sign_loss)]

cfg_select! {
    target_arch = "aarch64" => {
        pub mod aarch64;
        pub use aarch64 as arch;
    }
    target_arch = "x86_64" => {
        pub mod x86_64;
        pub mod x86;
        pub use x86_64 as arch;
    }
    target_arch = "x86" => {
        pub mod x86;
        pub use x86 as arch;
    }
    _ => {
        pub mod arch {
            pub use super::scalar_impl as seed_teams;
        }
    }
}

use std::{
    mem::transmute,
    ops::{Deref, DerefMut},
};

use crate::datatypes::{Index, Set};

pub use arch::seed_teams;

/// Mask for the initial seed portion of a packed seeding `u16`.
const INITIAL_SEED_MASK: u16 = 0x1F;

/// Return remaining team indices sorted by mid-stage seeding using scalar loops.
#[must_use]
pub fn scalar_impl(remaining: Set, diffs: &[i8; 16], opponents: &[Set; 16]) -> Seeding {
    let mut seeding = PackedSeeding::new();

    // Match only teams that remain in the tournament.
    for index in remaining {
        // Offset and invert signed values to embed an unsigned value into the packed integer.
        let diff = (15 - diffs[index.to_usize()]) as u16;
        let buchholz = (15
            - opponents[index.to_usize()]
                .into_iter()
                .map(|opponent| diffs[opponent.to_usize()])
                .sum::<i8>()) as u16;

        seeding[index.to_usize()] = diff << 10 | buchholz << 5 | index.to_u16();
    }

    seeding.sort_strip(remaining.len())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(align(32), C)]
pub struct PackedSeeding {
    data: [u16; 16],
}

impl PackedSeeding {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: [u16::MAX; _],
        }
    }

    #[must_use]
    pub fn sort_strip(mut self, len: usize) -> Seeding {
        self.data.sort_unstable();

        for packed_seed in &mut self[..len] {
            *packed_seed &= INITIAL_SEED_MASK;
        }

        // Safety: data is sorted and the active prefix has been masked out.
        unsafe { self.into_seeding_unchecked(len) }
    }

    /// # Safety
    ///
    /// Must be sorted with active prefixes stripped to construct a valid [`Seeding`].
    #[must_use]
    #[inline]
    pub unsafe fn into_seeding_unchecked(self, len: usize) -> Seeding {
        // `Index` is a transparent newtype of `u16`; the active prefix has been masked
        // down to only the initial seed, which is known to be in `0..16`.
        Seeding {
            len,
            data: unsafe { transmute::<[u16; 16], [Index; 16]>(self.data) },
        }
    }

    /// Returns a `*mut Self` which can be cast to other 8/16/32-byte aligned pointers, ie. SIMD vector types.
    pub const fn as_aligned_mut_ptr(&mut self) -> *mut Self {
        // `self.data` is 32-byte aligned.
        #[allow(clippy::cast_ptr_alignment)]
        self.data.as_mut_ptr().cast::<Self>()
    }
}

impl Default for PackedSeeding {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for PackedSeeding {
    type Target = [u16];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for PackedSeeding {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl From<[u16; 16]> for PackedSeeding {
    fn from(data: [u16; 16]) -> Self {
        Self { data }
    }
}

#[derive(Debug)]
pub struct SortedSeeding {
    data: [u16; 16],
}

impl SortedSeeding {
    #[must_use]
    pub fn with_len(self, len: usize) -> Seeding {
        // `Index` is a transparent newtype of `u16`; the active prefix has been masked
        // down to only the initial seed, which is known to be in `0..16`.
        Seeding {
            len,
            data: unsafe { transmute::<[u16; 16], [Index; 16]>(self.data) },
        }
    }
}

/// Sorted mid-stage seed order for the teams that remain in the tournament.
///
/// Only the first `len` entries are meaningful; deref exposes that active prefix
/// as a slice of team indices.
#[derive(Debug)]
pub struct Seeding {
    len: usize,
    data: [Index; 16],
}

impl Seeding {
    /// # Safety
    ///
    /// Must be sorted with active prefixes stripped to construct a valid [`Seeding`].
    #[must_use]
    #[inline]
    pub const unsafe fn from_indices_unchecked(len: usize, data: [Index; 16]) -> Self {
        Self { len, data }
    }

    /// Return the number of remaining teams in this seeding.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return whether there are no remaining teams in this seeding.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Deref for Seeding {
    type Target = [Index];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data[..self.len]
    }
}

impl PartialEq for Seeding {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (**self).eq(&(**other))
    }
}

impl Eq for Seeding {}

impl<I: std::slice::SliceIndex<[Index]>> std::ops::Index<I> for Seeding {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &(**self)[index]
    }
}

/// Lookup table for expanding opponent bits into byte masks.
///
/// Each 8-bit index maps to eight bytes. A set bit becomes `0xFF`; an unset
/// bit becomes `0x00`.
#[repr(align(64))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
struct ByteMasks([i64; 256]);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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
    use crate::datatypes::{Index, Set};

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

    type ImplFn = unsafe fn(Set, &[i8; 16], &[Set; 16]) -> super::Seeding;

    /// Tests that optimised implementations match the behaviour of the scalar implementation.
    fn assert_matches_scalar(implementation: &str, func: ImplFn) {
        for diffs in DIFF_FIXTURES {
            let opponents = std::array::from_fn(|i| {
                let rotate = u32::try_from(i).unwrap();
                set_from_bits(0b0001_0010_1010_0101_u16.rotate_left(rotate))
            });

            for bits in [
                0x0000, 0x0001, 0x8000, 0x00FF, 0xFF00, 0x5555, 0xAAAA, 0xFFFF,
            ] {
                let remaining = set_from_bits(bits);
                let expected = super::scalar_impl(remaining, &diffs, &opponents);
                let actual = unsafe { func(remaining, &diffs, &opponents) };

                assert_eq!(
                    &*expected, &*actual,
                    "{implementation} differs from scalar given fixture: {diffs:#?}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_feature = "neon")]
    fn neon_matches_scalar() {
        assert_matches_scalar("neon", super::aarch64::neon_impl);
    }

    #[test]
    #[cfg(target_feature = "sse2")]
    fn sse2_matches_scalar() {
        assert_matches_scalar("sse2", super::x86::sse2_impl);
    }

    #[test]
    #[cfg(target_feature = "avx2")]
    fn avx2_matches_scalar() {
        assert_matches_scalar("avx2", super::x86_64::avx2_impl);
    }
}
