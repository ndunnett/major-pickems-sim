//! Precalculate independent map win and best-of-three series win probabilities.
//!
//! The returned matrices are indexed as `[team_a][team_b]`; each cell stores the
//! probability that `team_a` beats `team_b` in that match format.
//!
//! A team wins the series by WW, WLW, or LWW.
//!
//! ```
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
    any(target_arch = "x86", target_arch = "x86_64") => {
        pub mod x86_64;
        pub use x86_64 as arch;
    }
    _ => {
        pub mod arch {
            pub use super::scalar_impl as calculate_probabilities;
        }
    }
}

use std::f32::consts::LOG2_10;

use crate::datatypes::{Rating, Sigma};

pub use arch::calculate_probabilities;

/// Precalculate BO1 and BO3 win-probability matrices.
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
