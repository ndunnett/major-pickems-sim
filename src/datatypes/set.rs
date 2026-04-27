use std::simd::Simd;

use crate::datatypes::Index;

/// Compact set of team indices backed by a 16-bit bitset.
///
/// This type is specialized for the simulator's fixed 16-team stages.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Set {
    data: u16,
}

impl Set {
    /// Construct a new empty set.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { data: 0 }
    }

    /// Construct a set containing all 16 valid indices.
    #[inline]
    #[must_use]
    pub const fn full() -> Self {
        Self { data: u16::MAX }
    }

    /// Insert an index into the set, returning whether the set changed.
    #[inline]
    pub const fn insert(&mut self, index: Index) -> bool {
        let old = self.data;
        self.data |= index.bit_select();
        old != self.data
    }

    /// Remove an index from the set, returning whether the set changed.
    #[inline]
    pub const fn remove(&mut self, index: Index) -> bool {
        let old = self.data;
        self.data &= !index.bit_select();
        old != self.data
    }

    /// Test whether the set contains an index.
    #[inline]
    #[must_use]
    pub const fn contains(self, index: Index) -> bool {
        (self.data & index.bit_select()) != 0
    }

    /// Test whether the set is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.data == 0
    }

    /// Return an iterator of contained indices in ascending order.
    #[inline]
    #[must_use]
    pub const fn iter(self) -> SetIter {
        SetIter { set: self }
    }

    /// Construct a SIMD vector with every lane set to the raw bitset value.
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

/// Iterator over indices contained in a [`Set`].
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
            // `trailing_zeros` is always less than 16 here because every stored
            // bit outside the low 16 bits is impossible for `u16`.
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

    macro_rules! i {
        ($n:expr) => {
            Index::new::<$n>()
        };
    }

    /// Compares functionality to `std::collections::HashSet`.
    #[test]
    fn compare_std_hashset() {
        use std::collections::HashSet;

        let samples = [
            vec![i!(0), i!(1), i!(2), i!(3)],
            vec![i!(15), i!(14), i!(13), i!(12)],
            vec![i!(0), i!(2), i!(4), i!(6), i!(8), i!(10), i!(12), i!(14)],
            vec![i!(1), i!(3), i!(5), i!(7), i!(9), i!(11), i!(13), i!(15)],
            Index::iter_all().collect(),
        ];

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
