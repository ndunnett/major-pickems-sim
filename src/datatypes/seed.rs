use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Initial tournament seed of a team.
///
/// Seeds are one-based and valid in the inclusive range `1..=16`.
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub struct Seed(u16);

impl Seed {
    pub fn try_new(seed: u16) -> anyhow::Result<Self> {
        if !(1..=16).contains(&seed) {
            anyhow::bail!("invalid seed: must be between 1 and 16");
        }

        Ok(Self(seed))
    }

    #[must_use]
    pub fn new(seed: u16) -> Self {
        Self::try_new(seed).unwrap()
    }

    /// Iterate through all valid initial seeds in ascending order.
    pub fn iter_all() -> impl Iterator<Item = Self> {
        (1..=16).map(|i| Self::try_new(i).unwrap())
    }
}

impl std::fmt::Display for Seed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Serialize for Seed {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u16(self.0)
    }
}

impl<'de> Deserialize<'de> for Seed {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::try_new(u16::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Zero-based index into arrays sorted by ascending initial seed.
///
/// `Index` is the simulation-facing companion to [`Seed`]: seed `1` maps to
/// index `0`, and seed `16` maps to index `15`.
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Index(u16);

impl Index {
    /// Construct an index from a compile-time constant.
    ///
    /// This fails to compile when `N >= 16`.
    #[inline]
    #[must_use]
    pub const fn new<const N: u16>() -> Self {
        const { assert!(N < 16, "invalid index value (must be less than 16)") };
        Self(N)
    }

    /// Construct an index from a runtime value.
    #[inline]
    pub fn try_new(n: u16) -> anyhow::Result<Self> {
        if n < 16 {
            Ok(Self(n))
        } else {
            Err(anyhow::anyhow!("invalid team index: {n}"))
        }
    }

    /// Return this index as a raw `u16`.
    #[inline]
    #[must_use]
    pub const fn to_u16(self) -> u16 {
        self.0
    }

    /// Return this index as a `usize` for array indexing.
    #[inline]
    #[must_use]
    pub const fn to_usize(self) -> usize {
        self.0 as usize
    }

    /// Convert this zero-based index into its one-based tournament seed.
    #[inline]
    #[must_use]
    pub fn to_seed(self) -> Seed {
        Seed::try_new(self.0 + 1).unwrap()
    }

    /// Return a bit mask selecting this index in a 16-bit [`Set`](crate::datatypes::Set).
    #[inline]
    #[must_use]
    pub const fn bit_select(self) -> u16 {
        1 << self.0
    }

    /// # Safety
    /// Must ensure that `n` < 16. This type is used to index 16 element arrays.
    #[inline]
    #[must_use]
    pub const unsafe fn from_u16(n: u16) -> Self {
        Self(n)
    }

    /// # Safety
    /// Must ensure that `n` < 16. This type is used to index 16 element arrays.
    #[inline]
    #[must_use]
    pub const unsafe fn from_u32(n: u32) -> Self {
        Self(n as u16)
    }

    /// # Safety
    /// Must ensure that `n` < 16. This type is used to index 16 element arrays.
    #[inline]
    #[must_use]
    pub const unsafe fn from_usize(n: usize) -> Self {
        Self(n as u16)
    }

    /// Iterate through every valid zero-based index.
    pub fn iter_all() -> impl Iterator<Item = Self> {
        (0..16).map(|i| unsafe { Self::from_u16(i) })
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Index> for Seed {
    fn from(index: Index) -> Self {
        // `Index` is guaranteed to be in 0..16, so adding one preserves the
        // `Seed` invariant of 1..=16.
        Self(index.0 + 1)
    }
}

impl From<Seed> for Index {
    fn from(seed: Seed) -> Self {
        // `Seed` is guaranteed to be in 1..=16, so subtracting one preserves the
        // `Index` invariant of 0..16.
        Self(seed.0 - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Input {
        seed: Seed,
    }

    #[test]
    fn validates_seeds() {
        assert!(Seed::try_new(1).is_ok());
        assert!(Seed::try_new(16).is_ok());
        assert!(Seed::try_new(0).is_err());
        assert!(Seed::try_new(17).is_err());
    }

    #[test]
    fn rejects_invalid_deserialized_seeds() {
        let input: Input = toml::from_str("seed = 16").unwrap();

        assert_eq!(input.seed.to_string(), "16");
        assert!(toml::from_str::<Input>("seed = 0").is_err());
        assert!(toml::from_str::<Input>("seed = 17").is_err());
    }
}
