use std::{cmp::Ordering, hash::Hash};

/// Team name used in TOML input and CLI pick arguments.
///
/// Names are trimmed, limited to 30 Unicode scalar values, and compared,
/// ordered, and hashed case-insensitively.
#[nutype::nutype(
    sanitize(trim),
    validate(not_empty, len_char_max = 30),
    derive(Debug, Display, Clone, Serialize, Deserialize)
)]
pub struct Name(String);

/// Case-insensitive name comparison.
impl PartialEq for Name {
    fn eq(&self, other: &Self) -> bool {
        // `nutype` stores the validated string transparently but does not expose
        // borrowed access, so this avoids allocating for the common exact-match
        // path before falling back to lowercase owned strings.
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

impl Eq for Name {}

/// Case-insensitive ordering.
impl Ord for Name {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut a = self.clone().into_inner();
        let mut b = other.clone().into_inner();
        a.make_ascii_lowercase();
        b.make_ascii_lowercase();
        a.cmp(&b)
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
        // Hashing must follow the same case-insensitive semantics as `Eq`.
        let s = unsafe { &*std::ptr::from_ref(self).cast::<String>() };

        for c in s.chars() {
            c.to_ascii_lowercase().hash(state);
        }
    }
}
