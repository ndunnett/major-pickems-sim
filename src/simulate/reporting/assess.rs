use std::{iter::Sum, ops::Add};

use crate::{
    data::TeamSeed,
    simulate::{Simulation, SwissSystem, TeamSet, reporting::Report},
};

/// Report for selecting optimal picks from basic statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssessReport {
    pub mean: f32,
    pub ds: f32,
    pub success: u64,
    pub n: u64,
    pub three_zero_picks: TeamSet,
    pub advanced_picks: TeamSet,
    pub zero_three_picks: TeamSet,
}

impl AssessReport {
    pub fn new<
        I1: IntoIterator<Item = TeamSeed>,
        I2: IntoIterator<Item = TeamSeed>,
        I3: IntoIterator<Item = TeamSeed>,
    >(
        three_zero_picks: I1,
        advanced_picks: I2,
        zero_three_picks: I3,
    ) -> Self {
        Self {
            mean: 0.0,
            ds: 0.0,
            success: 0,
            n: 0,
            three_zero_picks: three_zero_picks.into_iter().collect(),
            advanced_picks: advanced_picks.into_iter().collect(),
            zero_three_picks: zero_three_picks.into_iter().collect(),
        }
    }
}

impl Add for AssessReport {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        if self.n == 0 {
            rhs
        } else {
            self.success += rhs.success;

            let n_self = self.n as f32;
            let n_rhs = rhs.n as f32;
            let n = n_self + n_rhs;
            self.n += rhs.n;

            let delta = self.mean - rhs.mean;
            self.mean = (n_self * self.mean + n_rhs * rhs.mean) / n;
            self.ds += rhs.ds + delta * delta * (n_self * n_rhs / n);

            self
        }
    }
}

impl Sum for AssessReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, report| acc + report)
    }
}

impl Report for AssessReport {
    fn update(&mut self, ss: &SwissSystem) {
        // Update count
        self.n += 1;

        // Check results
        let stars = {
            self.three_zero_picks
                .iter()
                .filter(|&pick| ss.wins[pick as usize] == 3 && ss.losses[pick as usize] == 0)
                .count()
                + self
                    .advanced_picks
                    .iter()
                    .filter(|&pick| ss.wins[pick as usize] == 3 && ss.losses[pick as usize] > 0)
                    .count()
                + self
                    .zero_three_picks
                    .iter()
                    .filter(|&pick| ss.wins[pick as usize] == 0 && ss.losses[pick as usize] == 3)
                    .count()
        };

        // Update stats
        if stars >= 5 {
            self.success += 1;
        }

        let delta1 = stars as f32 - self.mean;
        self.mean += delta1 / self.n as f32;
        let delta2 = stars as f32 - self.mean;
        self.ds += delta1 * delta2;
    }

    fn format(&self, _sim: &Simulation) -> String {
        let mean = self.mean;
        let sd = (self.ds / self.n as f32).sqrt();
        let success = self.success as f32 / self.n as f32 * 100.0;

        format!(
            "\nSimulated stars earned: {mean:.3} +/- {sd:.3}\nExpected success (>=5 stars): {success:.1}%"
        )
    }
}
