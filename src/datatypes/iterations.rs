use std::{ops::Range, str::FromStr};

use rayon::iter::IntoParallelIterator;

/// Representation of simulation iterations.
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub struct Iterations(u64);

impl Iterations {
    pub fn try_new(iterations: u64) -> anyhow::Result<Self> {
        if iterations == 0 {
            anyhow::bail!("invalid iterations: must be greater than 0");
        }

        Ok(Self(iterations))
    }

    #[must_use]
    pub fn new(iterations: u64) -> Self {
        Self::try_new(iterations).unwrap()
    }

    /// # Safety
    /// Must ensure that `iterations` is greater than zero.
    #[must_use]
    pub const unsafe fn new_unchecked(iterations: u64) -> Self {
        Self(iterations)
    }

    #[must_use]
    pub const fn to_u64(self) -> u64 {
        self.0
    }

    #[must_use]
    pub const fn to_f32(self) -> f32 {
        self.0 as f32
    }
}

impl Default for Iterations {
    fn default() -> Self {
        Self(1_000_000)
    }
}

impl TryFrom<u64> for Iterations {
    type Error = anyhow::Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::try_new(value)
    }
}

impl FromStr for Iterations {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_new(s.parse()?)
    }
}

impl std::fmt::Display for Iterations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl IntoIterator for Iterations {
    type Item = u64;
    type IntoIter = Range<u64>;

    fn into_iter(self) -> Self::IntoIter {
        0..self.0
    }
}

impl IntoParallelIterator for Iterations {
    type Item = u64;
    type Iter = rayon::range::Iter<u64>;

    fn into_par_iter(self) -> Self::Iter {
        (0..self.0).into_par_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_iterationss() {
        assert!(Iterations::try_new(1).is_ok());
        assert!(Iterations::try_new(0).is_err());
        assert!(Iterations::from_str("1").is_ok());
        assert!(Iterations::from_str("0").is_err());
    }
}
