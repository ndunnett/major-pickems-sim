use std::simd::Simd;

use crate::datatypes::Index;

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
    pub const fn insert(&mut self, index: Index) -> bool {
        let old = self.data;
        self.data |= index.bit_select();
        old != self.data
    }

    /// Removes index from the set.
    #[inline]
    pub const fn remove(&mut self, index: Index) -> bool {
        let old = self.data;
        self.data &= !index.bit_select();
        old != self.data
    }

    /// Tests if the set contains index.
    #[inline]
    #[must_use]
    pub const fn contains(self, index: Index) -> bool {
        (self.data & index.bit_select()) != 0
    }

    /// Tests if the set is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.data == 0
    }

    /// Returns an iterator of `Index`s (copied) contained within the set.
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

impl FromIterator<Index> for Set {
    fn from_iter<I: IntoIterator<Item = Index>>(indices: I) -> Self {
        let mut set = Self::new();

        for index in indices {
            set.insert(index);
        }

        set
    }
}

/// Struct to iterate `Index`s contained within a `TeamSet`.
pub struct SetIter {
    set: Set,
}

impl Iterator for SetIter {
    type Item = Index;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.set.is_empty() {
            None
        } else {
            let next = unsafe { Index::from_u32(self.set.data.trailing_zeros()) };
            self.set.data &= self.set.data - 1;
            Some(next)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::datatypes::Index;

    /// Compares functionality to `std::collections::HashSet`.
    #[test]
    fn compare_std_hashset() {
        use std::collections::HashSet;

        let samples = unsafe {
            [
                vec![
                    Index::from_u16(0),
                    Index::from_u16(1),
                    Index::from_u16(2),
                    Index::from_u16(3),
                ],
                vec![
                    Index::from_u16(15),
                    Index::from_u16(14),
                    Index::from_u16(13),
                    Index::from_u16(12),
                ],
                vec![
                    Index::from_u16(0),
                    Index::from_u16(2),
                    Index::from_u16(4),
                    Index::from_u16(6),
                    Index::from_u16(8),
                    Index::from_u16(10),
                    Index::from_u16(12),
                    Index::from_u16(14),
                ],
                vec![
                    Index::from_u16(1),
                    Index::from_u16(3),
                    Index::from_u16(5),
                    Index::from_u16(7),
                    Index::from_u16(9),
                    Index::from_u16(11),
                    Index::from_u16(13),
                    Index::from_u16(15),
                ],
                vec![
                    Index::from_u16(0),
                    Index::from_u16(1),
                    Index::from_u16(2),
                    Index::from_u16(3),
                    Index::from_u16(4),
                    Index::from_u16(5),
                    Index::from_u16(6),
                    Index::from_u16(7),
                    Index::from_u16(8),
                    Index::from_u16(9),
                    Index::from_u16(10),
                    Index::from_u16(11),
                    Index::from_u16(12),
                    Index::from_u16(13),
                    Index::from_u16(14),
                    Index::from_u16(15),
                ],
            ]
        };

        for sample in samples {
            let mut reference = HashSet::new();
            let mut set = Set::new();

            assert_eq!(set.is_empty(), reference.is_empty());

            for i in &sample {
                assert_eq!(set.insert(*i), reference.insert(*i));
            }

            for i in (0..16).map(|n| unsafe { Index::from_u16(n) }) {
                assert_eq!(set.contains(i), reference.contains(&i));
            }

            assert_eq!(set.iter().collect::<HashSet<_>>(), reference);
            assert_eq!(set.is_empty(), reference.is_empty());

            for i in &sample {
                assert_eq!(set.remove(*i), reference.remove(i));
            }

            for i in (0..16).map(|n| unsafe { Index::from_u16(n) }) {
                assert_eq!(set.contains(i), reference.contains(&i));
            }

            assert_eq!(set.is_empty(), reference.is_empty());
        }
    }
}
