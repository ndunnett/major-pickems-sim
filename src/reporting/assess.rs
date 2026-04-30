use std::{iter::Sum, ops::Add};

use anyhow::anyhow;

use crate::{
    datatypes::{Index, Name, Set, Teams},
    reporting::Report,
    simulation::{Simulation, SwissSystem},
};

/// Report for estimating how often a concrete pick set earns enough stars.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssessReport {
    /// Running mean of stars earned.
    pub mean: f32,
    /// Sum of squared differences for variance calculation.
    pub ds: f32,
    /// Number of simulations with at least five stars.
    pub success: u64,
    /// Number of simulated tournaments assessed.
    pub n: u64,
    /// Teams selected as 3-0 picks.
    pub three_zero_picks: Set,
    /// Teams selected as 3-1/3-2 advancement picks.
    pub advancing_picks: Set,
    /// Teams selected as 0-3 picks.
    pub zero_three_picks: Set,
}

impl AssessReport {
    /// Construct an assessment report from selected team indices.
    pub fn new<
        I1: IntoIterator<Item = Index>,
        I2: IntoIterator<Item = Index>,
        I3: IntoIterator<Item = Index>,
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
            advancing_picks: advanced_picks.into_iter().collect(),
            zero_three_picks: zero_three_picks.into_iter().collect(),
        }
    }

    /// Resolve CLI team-name picks into simulation indices.
    pub fn try_from_args(
        teams: &Teams,
        three_zero_str: &[Name; 2],
        advancing_str: &[Name; 6],
        zero_three_str: &[Name; 2],
    ) -> anyhow::Result<Self> {
        let mut three_zero_picks = Vec::new();
        let mut advancing_picks = Vec::new();
        let mut zero_three_picks = Vec::new();

        for s in three_zero_str {
            three_zero_picks.push(Index::try_new(
                teams
                    .names
                    .iter()
                    .position(|name| name == s)
                    .ok_or_else(|| anyhow!("failed to find team \"{s}\" in the input file"))?
                    as u16,
            )?);
        }

        for s in advancing_str {
            advancing_picks.push(Index::try_new(
                teams
                    .names
                    .iter()
                    .position(|name| name == s)
                    .ok_or_else(|| anyhow!("failed to find team \"{s}\" in the input file"))?
                    as u16,
            )?);
        }

        for s in zero_three_str {
            zero_three_picks.push(Index::try_new(
                teams
                    .names
                    .iter()
                    .position(|name| name == s)
                    .ok_or_else(|| anyhow!("failed to find team \"{s}\" in the input file"))?
                    as u16,
            )?);
        }

        let mut seen = Set::new();

        for pick in three_zero_picks
            .iter()
            .chain(advancing_picks.iter())
            .chain(zero_three_picks.iter())
        {
            if !seen.insert(*pick) {
                anyhow::bail!("duplicate pick: {}", teams.names[pick.to_usize()]);
            }
        }

        Ok(Self::new(
            three_zero_picks,
            advancing_picks,
            zero_three_picks,
        ))
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
            self.mean = n_rhs.mul_add(rhs.mean, n_self * self.mean) / n;
            self.ds += (delta * delta).mul_add(n_self * n_rhs / n, rhs.ds);

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
        // Update count before Welford's mean/variance step.
        self.n += 1;

        // Count stars under the current pick'em rules represented by this
        // report: exact 3-0, non-3-0 advancement, and exact 0-3.
        let stars = {
            self.three_zero_picks
                .iter()
                .filter(|&pick| ss.wins[pick.to_usize()] == 3 && ss.losses[pick.to_usize()] == 0)
                .count()
                + self
                    .advancing_picks
                    .iter()
                    .filter(|&pick| ss.wins[pick.to_usize()] == 3 && ss.losses[pick.to_usize()] > 0)
                    .count()
                + self
                    .zero_three_picks
                    .iter()
                    .filter(|&pick| {
                        ss.wins[pick.to_usize()] == 0 && ss.losses[pick.to_usize()] == 3
                    })
                    .count()
        };

        // Update success count and running distribution of stars.
        if stars >= 5 {
            self.success += 1;
        }

        let delta1 = stars as f32 - self.mean;
        self.mean += delta1 / self.n as f32;
        let delta2 = stars as f32 - self.mean;
        self.ds = delta1.mul_add(delta2, self.ds);
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
