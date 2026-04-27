use std::simd::Simd;

use crate::datatypes::Seed;

/// High performance set, specifically for teams.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Set {
    data: u16,
}

impl Set {
    /// Constructs a new empty set.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { data: 0 }
    }

    /// Constructs a new full set.
    #[inline]
    #[must_use]
    pub const fn full() -> Self {
        Self { data: u16::MAX }
    }

    /// Inserts index into the set.
    #[inline]
    pub const fn insert(&mut self, index: Seed) -> bool {
        let old = self.data;
        self.data |= 1 << index;
        old != self.data
    }

    /// Removes index from the set.
    #[inline]
    pub const fn remove(&mut self, index: Seed) -> bool {
        let old = self.data;
        self.data &= !(1 << index);
        old != self.data
    }

    /// Tests if the set contains index.
    #[inline]
    #[must_use]
    pub const fn contains(self, index: Seed) -> bool {
        (self.data & (1 << index)) != 0
    }

    /// Tests if the set is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.data == 0
    }

    /// Returns an iterator of `Seed`s (copied) contained within the set.
    #[inline]
    #[must_use]
    pub const fn iter(self) -> SetIter {
        SetIter { set: self }
    }

    // Constructs a new SIMD vector with all elements set to the current state of the set.
    #[inline]
    #[must_use]
    pub const fn splat<const N: usize>(self) -> Simd<u16, N> {
        Simd::splat(self.data)
    }
}

impl Default for Set {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl FromIterator<Seed> for Set {
    fn from_iter<I: IntoIterator<Item = Seed>>(seeds: I) -> Self {
        let mut set = Self::new();

        for seed in seeds {
            set.insert(seed);
        }

        set
    }
}

/// Struct to iterate `Seed`s contained within a `TeamSet`.
pub struct SetIter {
    set: Set,
}

impl Iterator for SetIter {
    type Item = Seed;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.set.is_empty() {
            None
        } else {
            let next = self.set.data.trailing_zeros() as Seed;
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
            let mut set = Set::new();

            assert_eq!(set.is_empty(), reference.is_empty());

            for i in &sample {
                assert_eq!(set.insert(*i), reference.insert(*i));
            }

            for i in 0..16 {
                assert_eq!(set.contains(i), reference.contains(&i));
            }

            assert_eq!(set.iter().collect::<HashSet<_>>(), reference);
            assert_eq!(set.is_empty(), reference.is_empty());

            for i in &sample {
                assert_eq!(set.remove(*i), reference.remove(i));
            }

            for i in 0..16 {
                assert_eq!(set.contains(i), reference.contains(&i));
            }

            assert_eq!(set.is_empty(), reference.is_empty());
        }
    }
}
