/// Represents the initial seed of a team.
#[nutype::nutype(
    validate(greater_or_equal = 1, less_or_equal = 16),
    derive(
        Debug,
        Display,
        Clone,
        Copy,
        Serialize,
        Deserialize,
        PartialOrd,
        Ord,
        PartialEq,
        Eq
    )
)]
pub struct Seed(u16);

impl Seed {
    pub fn increment(self) -> Result<Self, SeedError> {
        let n = unsafe { std::mem::transmute::<Self, u16>(self) };
        Self::try_new(n + 1)
    }

    pub fn iter_all() -> impl Iterator<Item = Self> {
        (1..=16).map(|i| Self::try_new(i).unwrap())
    }
}

/// Represents the index of a team within an array sorted in ascending order by initial seed.
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Index(u16);

impl Index {
    #[inline]
    #[must_use]
    pub const fn new<const N: u16>() -> Self {
        const { assert!(N < 16, "invalid index value (must be less than 16)") };
        Self(N)
    }

    #[inline]
    pub fn try_new(n: u16) -> anyhow::Result<Self> {
        if n < 16 {
            Ok(Self(n))
        } else {
            Err(anyhow::anyhow!("invalid team index: {n}"))
        }
    }

    #[inline]
    #[must_use]
    pub const fn to_usize(self) -> usize {
        self.0 as usize
    }

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

    pub fn iter_all() -> impl Iterator<Item = Self> {
        (0..16).map(|i| unsafe { Self::from_u16(i) })
    }
}

impl From<Seed> for Index {
    fn from(seed: Seed) -> Self {
        Self(unsafe { std::mem::transmute::<Seed, u16>(seed) } - 1)
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
