//! Precalculate independent map win and best-of-three series win probabilities.
//!
//! The returned matrices are indexed as `[team_a][team_b]`; each cell stores the
//! probability that `team_a` beats `team_b` in that match format.
//!
//! A team wins the series by WW, WLW, or LWW.
//!
//! ```text
//! where:
//!     Ra = team A rating
//!     Rb = team B rating
//!     P = team A map win probability
//!
//! P = 1 / (1 + 10^((Rb - Ra) / sigma))
//!
//! substitute:
//!     10^x => 2^(log2(10) * x)
//!
//! P = 1 / (1 + 2^(log2(10) * (Rb - Ra) / sigma))
//!
//! where:
//!     u = log2(10) / sigma
//!     v = (Rb - Ra) * u
//!     w = 2^v
//!
//! [eq. 1] map win probability: P = 1 / (1 + w)
//!
//! where:
//!     Q = team A series win probability
//!     team A wins the series by WW, WLW, or LWW
//!
//! Q(W) = P
//! Q(L) = 1 - P
//! Q(WW-) = Q(W) * Q(W)
//!        = P * P
//! Q(WLW) = Q(LWW) = Q(W) * Q(W) * Q(L)
//!                 = P * P * (1 - P)
//! Q = Q(WLW) + Q(LWW) + Q(WW-)
//!   = 2 * (P * P * (1 - P)) + P * P
//!
//! where:
//!     a = P * P
//!     b = 1 - P
//!
//! [eq. 2] series win probability: Q = 2 * a * b + a
//! ```

#![allow(clippy::many_single_char_names)]

cfg_select! {
    target_arch = "aarch64" => {
        pub mod aarch64;
        pub use aarch64 as arch;
    }
    target_arch = "x86_64" => {
        pub mod x86_64;
        pub mod x86;
        pub use x86_64 as arch;
    }
    target_arch = "x86" => {
        pub mod x86;
        pub use x86 as arch;
    }
    _ => {
        pub mod arch {
            pub use super::scalar_impl as calculate_probabilities;
        }
    }
}

use std::f32::consts::LOG2_10;

/// Constants used for `exp2` approximations.
mod exp2_constants {
    pub const POLY_0: f32 = 0.000_198_756_91;
    pub const POLY_1: f32 = 0.001_398_199_9;
    pub const POLY_2: f32 = 0.008_333_452;
    pub const POLY_3: f32 = 0.041_665_796;
    pub const POLY_4: f32 = 0.166_666_66;
    pub const POLY_5: f32 = 0.500_000_06;
    pub const INPUT_MIN: f32 = -126.0;
    pub const INPUT_MAX: f32 = 127.0;
    pub const EXPONENT_BIAS: i32 = 0x7f;
    pub const MANTISSA_BITS: i32 = 23;
}

use crate::datatypes::{Rating, Sigma};

pub use arch::calculate_probabilities;

/// Precalculate BO1 and BO3 win-probability matrices using scalar loops.
///
/// The result is `[probabilities_bo1, probabilities_bo3]`, and each matrix is
/// indexed by `[team_a][team_b]`.
#[must_use]
pub fn scalar_impl(ratings: [Rating; 16], sigma: Sigma) -> [[[f32; 16]; 16]; 2] {
    // u = log2(10) / sigma
    // v = (Rb - Ra) * u
    // w = 2^v
    // P = 1 / (1 + w)
    let u = LOG2_10 / sigma.to_f32();
    let mut probabilities_bo1 = [[0.5; 16]; 16];

    for (i, ra) in ratings.iter().enumerate() {
        let ra = ra.to_f32();

        for (j, rb) in ratings.iter().enumerate().skip(i + 1) {
            let rb = rb.to_f32();
            let v = (rb - ra) * u;
            let w = v.exp2();
            let p = (1.0 + w).recip();
            probabilities_bo1[i][j] = p;
            probabilities_bo1[j][i] = 1.0 - p;
        }
    }

    // a = P * P
    // b = 1 - P
    // Q = 2 * a * b + a
    let mut probabilities_bo3 = [[0.5; 16]; 16];

    for i in 0..16 {
        for j in i + 1..16 {
            let p = probabilities_bo1[i][j];
            let a = p * p;
            let b = 1.0 - p;
            let q = 2.0f32.mul_add(a * b, a);
            probabilities_bo3[i][j] = q;
            probabilities_bo3[j][i] = 1.0 - q;
        }
    }

    [probabilities_bo1, probabilities_bo3]
}

#[cfg(test)]
mod tests {
    use crate::datatypes::{Rating, Sigma};

    const TOLERANCE: f32 = 1e-5;
    const SIGMAS: [Sigma; 2] =
        unsafe { [Sigma::new_unchecked(800.0), Sigma::new_unchecked(400.0)] };
    const RATING_FIXTURES: [[u16; 16]; 3] = [
        [
            1_100, 1_180, 1_260, 1_340, 1_420, 1_500, 1_580, 1_660, 1_740, 1_820, 1_900, 1_980,
            2_060, 2_140, 2_220, 2_300,
        ],
        [
            1_850, 1_120, 2_040, 1_530, 1_760, 2_270, 1_390, 1_970, 1_210, 2_130, 1_680, 1_470,
            2_360, 1_590, 1_910, 1_300,
        ],
        [
            1_600, 1_600, 1_610, 1_610, 1_620, 1_620, 1_630, 1_630, 1_640, 1_640, 1_650, 1_650,
            1_660, 1_660, 1_670, 1_670,
        ],
    ];

    type ImplFn = unsafe fn([Rating; 16], Sigma) -> [[[f32; 16]; 16]; 2];

    /// Tests that optimised implementations match the behaviour of the scalar implementation.
    fn assert_matches_scalar(implementation: &str, func: ImplFn) {
        for fixture in RATING_FIXTURES {
            let ratings = fixture.map(Rating::new);

            for sigma in SIGMAS {
                let expected = super::scalar_impl(ratings, sigma);
                let actual = unsafe { func(ratings, sigma) };

                for matrix in 0..2 {
                    for row in 0..16 {
                        for column in 0..16 {
                            let expected = expected[matrix][row][column];
                            let actual = actual[matrix][row][column];
                            let difference = (expected - actual).abs();

                            assert!(
                                difference <= TOLERANCE,
                                "{implementation} differs from scalar at matrix {matrix}, row {row}, column {column}: expected {expected}, actual {actual}, difference {difference}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(target_feature = "neon")]
    fn neon_matches_scalar() {
        assert_matches_scalar("neon", super::aarch64::neon_impl);
    }

    #[test]
    #[cfg(target_feature = "sse2")]
    fn sse2_matches_scalar() {
        assert_matches_scalar("sse2", super::x86::sse2_impl);
    }

    #[test]
    #[cfg(target_feature = "avx2")]
    fn avx2_matches_scalar() {
        assert_matches_scalar("avx2", super::x86_64::avx2_impl);
    }
}
