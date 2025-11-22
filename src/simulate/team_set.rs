use std::simd::{LaneCount, Simd, SupportedLaneCount};

use crate::data::TeamSeed;

/// High performance set, specifically for teams.
#[derive(Clone, Copy, PartialEq)]
pub struct TeamSet {
    data: u16,
}

impl TeamSet {
    /// Constructs a new empty set.
    #[inline(always)]
    pub const fn new() -> Self {
        Self { data: 0 }
    }

    /// Constructs a new full set.
    #[inline(always)]
    pub const fn full() -> Self {
        Self { data: u16::MAX }
    }

    /// Inserts index into the set.
    #[inline(always)]
    pub fn insert(&mut self, index: TeamSeed) -> bool {
        let old = self.data;
        self.data |= 1 << index;
        old != self.data
    }

    /// Removes index from the set.
    #[inline(always)]
    pub fn remove(&mut self, index: &TeamSeed) -> bool {
        let old = self.data;
        self.data &= !(1 << index);
        old != self.data
    }

    /// Tests if the set contains index.
    #[inline(always)]
    pub fn contains(&self, index: &TeamSeed) -> bool {
        (self.data & (1 << index)) != 0
    }

    /// Tests if the set is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data == 0
    }

    /// Returns an iterator of `TeamSeed`s (copied) contained within the set.
    #[inline(always)]
    pub fn iter(&self) -> TeamSetIter {
        TeamSetIter { set: *self }
    }

    // Constructs a new SIMD vector with all elements set to the current state of the set.
    #[inline(always)]
    pub fn splat<const N: usize>(&self) -> Simd<u16, N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        Simd::splat(self.data)
    }
}

impl Default for TeamSet {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TeamSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl FromIterator<TeamSeed> for TeamSet {
    fn from_iter<I: IntoIterator<Item = TeamSeed>>(seeds: I) -> Self {
        let mut set = Self::new();

        for seed in seeds {
            set.insert(seed);
        }

        set
    }
}

/// Struct to iterate `TeamSeed`s contained within a `TeamSet`.
pub struct TeamSetIter {
    set: TeamSet,
}

impl Iterator for TeamSetIter {
    type Item = TeamSeed;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.set.is_empty() {
            None
        } else {
            let next = self.set.data.trailing_zeros() as TeamSeed;
            self.set.data &= self.set.data - 1;
            Some(next)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compares functionality to `std::collections::HashSet`.
    #[test]
    fn compare_std_hashset() {
        use std::collections::HashSet;

        let samples = [
            vec![0, 1, 2, 3],
            vec![15, 14, 13, 12],
            vec![0, 2, 4, 6, 8, 10, 12, 14],
            vec![1, 3, 5, 7, 9, 11, 13, 15],
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ];

        for sample in samples {
            let mut reference = HashSet::new();
            let mut set = TeamSet::new();

            assert_eq!(set.is_empty(), reference.is_empty());

            for i in &sample {
                assert_eq!(set.insert(*i), reference.insert(*i));
            }

            for i in 0..16 {
                assert_eq!(set.contains(&i), reference.contains(&i));
            }

            assert_eq!(set.iter().collect::<HashSet<_>>(), reference);
            assert_eq!(set.is_empty(), reference.is_empty());

            for i in &sample {
                assert_eq!(set.remove(i), reference.remove(i));
            }

            for i in 0..16 {
                assert_eq!(set.contains(&i), reference.contains(&i));
            }

            assert_eq!(set.is_empty(), reference.is_empty());
        }
    }
}
