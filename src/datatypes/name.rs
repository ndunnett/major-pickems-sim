use std::hash::Hash;

/// Represents a team name.
#[nutype::nutype(
    sanitize(trim),
    validate(not_empty, len_char_max = 30),
    derive(Debug, Display, Clone, Serialize, Deserialize, PartialOrd, Ord, Eq)
)]
pub struct Name(String);

/// Case insensitive name comparison.
impl PartialEq for Name {
    fn eq(&self, other: &Self) -> bool {
        let a = unsafe { &*std::ptr::from_ref(self).cast::<String>() };
        let b = unsafe { &*std::ptr::from_ref(other).cast::<String>() };

        if a == b {
            return true;
        }

        let mut a = self.clone().into_inner();
        let mut b = other.clone().into_inner();
        a.make_ascii_lowercase();
        b.make_ascii_lowercase();
        a == b
    }
}

/// Case insensitive hashing.
impl Hash for Name {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let s = unsafe { &*std::ptr::from_ref(self).cast::<String>() };

        for c in s.chars() {
            c.to_ascii_lowercase().hash(state);
        }
    }
}
