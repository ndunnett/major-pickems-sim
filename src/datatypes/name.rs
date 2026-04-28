use std::{cmp::Ordering, hash::Hash};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Team name used in TOML input and CLI pick arguments.
///
/// Names are trimmed, limited to 30 characters, and compared,
/// ordered, and hashed case-insensitively.
#[derive(Debug, Clone)]
pub struct Name(String);

impl Name {
    pub fn try_new(name: impl AsRef<str>) -> anyhow::Result<Self> {
        let name = name.as_ref().trim();

        if name.is_empty() {
            anyhow::bail!("invalid name: cannot be empty");
        }

        if name.chars().count() > 30 {
            anyhow::bail!("invalid name: cannot be longer than 30 characters");
        }

        Ok(Self(name.to_string()))
    }

    pub fn new(name: impl AsRef<str>) -> Self {
        Self::try_new(name).unwrap()
    }

    /// # Safety
    /// Must ensure that `name` is less than 30 characters and has whitespace trimmed.
    #[inline]
    #[must_use]
    pub const unsafe fn new_unchecked(name: String) -> Self {
        Self(name)
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for Name {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Name {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::try_new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Case-insensitive name comparison.
impl PartialEq for Name {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_ignore_ascii_case(&other.0)
    }
}

impl Eq for Name {}

/// Case-insensitive ordering.
impl Ord for Name {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .chars()
            .map(|c| c.to_ascii_lowercase())
            .cmp(other.0.chars().map(|c| c.to_ascii_lowercase()))
    }
}

impl PartialOrd for Name {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Case-insensitive hashing.
impl Hash for Name {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for c in self.0.chars() {
            c.to_ascii_lowercase().hash(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_trims_names() {
        assert_eq!(Name::new("  Vitality  ").to_string(), "Vitality");
        assert!(Name::try_new("   ").is_err());
        assert!(Name::try_new("1234567890123456789012345678901").is_err());
    }

    #[test]
    fn compares_and_hashes_case_insensitively() {
        use std::collections::{BTreeSet, HashSet};

        let name = Name::new("NAVI");
        let lowercase = Name::new("navi");

        assert_eq!(name, lowercase);

        let mut hash_set = HashSet::new();
        hash_set.insert(name.clone());
        assert!(hash_set.contains(&lowercase));

        let mut tree_set = BTreeSet::new();
        tree_set.insert(name);
        tree_set.insert(lowercase);
        assert_eq!(tree_set.len(), 1);
    }
}
