//! Fast sort and strip implementations specifically for `PackedSeeding`.
//!
//! `PackedSeeding` is a 32-byte aligned `[u16; 16]` to represent the packed seeding
//! information, using `u16::MAX` as a sentinel for inactive teams. The underlying
//! array must be sorted and then masked to eliminate the packed information to leave
//! only the initial seed value.

cfg_select! {
    target_arch = "aarch64" => {
        pub mod aarch64;
    }
    target_arch = "x86_64" => {
        pub mod x86_64;
        pub mod x86;
    }
    target_arch = "x86" => {
        pub mod x86;
    }
    _ => {}
}

#[cfg(test)]
mod tests {
    use crate::simulation::seeding::{PackedSeeding, Seeding};

    type ImplFn = unsafe fn(PackedSeeding, usize) -> Seeding;

    fn assert_sorting_correct(implementation: &str, func: ImplFn) {
        for len in [4, 6, 8, 12, 16] {
            for bits in u16::MIN..=u16::MAX {
                let case = PackedSeeding::from(std::array::from_fn(|i| {
                    if i < len {
                        u16::from(bits & (1 << i) != 0)
                    } else {
                        1
                    }
                }));

                let actual = unsafe { func(case, len) };
                let expected = case.sort_strip(len);

                assert_eq!(
                    expected, actual,
                    "{implementation} failed for bitset {bits:#06X}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_feature = "neon")]
    fn neon_correctness() {
        assert_sorting_correct("neon", super::aarch64::sort_strip_neon);
    }

    #[test]
    #[cfg(target_feature = "sse2")]
    fn sse2_correctness() {
        assert_sorting_correct("sse2", super::x86::sort_strip_sse2);
    }

    #[test]
    #[cfg(target_feature = "avx2")]
    fn avx2_correctness() {
        assert_sorting_correct("avx2", super::x86_64::sort_strip_avx2);
    }
}
