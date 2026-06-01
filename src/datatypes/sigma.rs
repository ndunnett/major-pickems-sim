use std::str::FromStr;

/// Representation of simulation sigma.
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct Sigma(f32);

impl Sigma {
    pub fn try_new(sigma: f32) -> anyhow::Result<Self> {
        if !sigma.is_sign_positive() || !sigma.is_normal() {
            anyhow::bail!("invalid sigma: must be greater than 0");
        }

        Ok(Self(sigma))
    }

    #[must_use]
    pub fn new(sigma: f32) -> Self {
        Self::try_new(sigma).unwrap()
    }

    /// # Safety
    /// Must ensure that `sigma` is greater than zero.
    #[must_use]
    pub const unsafe fn new_unchecked(sigma: f32) -> Self {
        Self(sigma)
    }

    #[must_use]
    pub const fn to_f32(self) -> f32 {
        self.0
    }
}

impl Default for Sigma {
    fn default() -> Self {
        Self(800.0)
    }
}

impl TryFrom<f32> for Sigma {
    type Error = anyhow::Error;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::try_new(value)
    }
}

impl FromStr for Sigma {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_new(s.parse()?)
    }
}

impl std::fmt::Display for Sigma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_sigmas() {
        assert!(Sigma::try_new(800.0).is_ok());
        assert!(Sigma::try_new(0.0).is_err());
        assert!(Sigma::try_new(f32::NAN).is_err());
        assert!(Sigma::try_new(f32::INFINITY).is_err());
        assert!(Sigma::from_str("800").is_ok());
        assert!(Sigma::from_str("0").is_err());
    }
}
