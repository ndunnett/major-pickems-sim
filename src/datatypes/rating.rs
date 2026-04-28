use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Representation of team rating.
#[derive(Debug, Clone, Copy)]
pub struct Rating(u16);

impl Rating {
    pub fn try_new(rating: u16) -> anyhow::Result<Self> {
        if rating == 0 {
            anyhow::bail!("invalid rating: must be greater than 0");
        }

        Ok(Self(rating))
    }

    #[must_use]
    pub fn new(rating: u16) -> Self {
        Self::try_new(rating).unwrap()
    }

    /// # Safety
    /// Must ensure that `rating` is greater than zero.
    #[must_use]
    pub const unsafe fn new_unchecked(rating: u16) -> Self {
        Self(rating)
    }

    /// Return the rating as `f32` for probability calculations.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from(self.0)
    }
}

impl std::fmt::Display for Rating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Serialize for Rating {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u16(self.0)
    }
}

impl<'de> Deserialize<'de> for Rating {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::try_new(u16::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Input {
        rating: Rating,
    }

    #[test]
    fn validates_ratings() {
        assert!(Rating::try_new(1).is_ok());
        assert!(Rating::try_new(0).is_err());
    }

    #[test]
    fn rejects_invalid_deserialized_ratings() {
        let input: Input = toml::from_str("rating = 2000").unwrap();
        assert!((input.rating.to_f32() - 2000.0).abs() < f32::EPSILON);
        assert!(toml::from_str::<Input>("rating = 0").is_err());
    }
}
