use std::{iter::Sum, ops::Add};

use crate::{
    datatypes::Rating,
    reporting::Report,
    simulation::{Simulation, SwissSystem},
};

/// Running distribution statistics for one team.
#[derive(Debug, Clone, Copy, Default)]
pub struct DistributionStats {
    /// Mean opponent rating.
    pub r_mean: f32,
    /// Sum of squared rating differences for variance calculation.
    pub r_ds: f32,
    /// Mean BO1 win probability against opponents faced.
    pub p_mean: f32,
    /// Sum of squared probability differences for variance calculation.
    pub p_ds: f32,
    /// Number of opponent observations.
    pub n: u64,
}

/// Report to record relative strength of opponents faced for each team.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrengthReport {
    /// Per-team opponent strength statistics, indexed by initial seed index.
    pub stats: [DistributionStats; 16],
}

impl Add for StrengthReport {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (seed, a) in self.stats.iter_mut().enumerate() {
            let b = rhs.stats[seed];

            if a.n == 0 {
                *a = rhs.stats[seed];
            } else {
                let na = a.n as f32;
                let nb = b.n as f32;
                let n = na + nb;

                // Merge two Welford accumulators for opponent rating stats.
                let r_delta = a.r_mean - b.r_mean;
                a.r_mean = nb.mul_add(b.r_mean, na * a.r_mean) / n;
                a.r_ds += (r_delta * r_delta).mul_add(na * nb / n, b.r_ds);

                // Merge two Welford accumulators for win-probability stats.
                let p_delta = a.p_mean - b.p_mean;
                a.p_mean = nb.mul_add(b.p_mean, na * a.p_mean) / n;
                a.p_ds += (p_delta * p_delta).mul_add(na * nb / n, b.p_ds);

                // Combine count after both means use the previous counts.
                a.n += b.n;
            }
        }

        self
    }
}

impl Sum for StrengthReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, report| acc + report)
    }
}

impl Report for StrengthReport {
    fn update(&mut self, ss: &SwissSystem) {
        for (seed, result) in self.stats.iter_mut().enumerate() {
            for opponent in ss.opponents[seed].iter() {
                // Welford's online algorithm keeps variance stable while
                // accumulating many parallel simulation samples.
                result.n += 1;

                // Update rating distribution.
                let r = ss.ratings[opponent.to_usize()].to_f32();
                let r_delta1 = r - result.r_mean;
                result.r_mean += r_delta1 / result.n as f32;
                let r_delta2 = r - result.r_mean;
                result.r_ds = r_delta1.mul_add(r_delta2, result.r_ds);

                // Update distribution of this team's BO1 win probabilities
                // against the opponents it actually drew.
                let p = ss.probabilities_bo1[seed][opponent.to_usize()] * 100.0;
                let p_delta1 = p - result.p_mean;
                result.p_mean += p_delta1 / result.n as f32;
                let p_delta2 = p - result.p_mean;
                result.p_ds = p_delta1.mul_add(p_delta2, result.p_ds);
            }
        }
    }

    fn format(&self, sim: &Simulation) -> String {
        let mut out = Vec::new();

        // Get mean rating of teams
        let baseline = sim
            .teams
            .ratings
            .iter()
            .copied()
            .map(Rating::to_f32)
            .sum::<f32>()
            / 16.0;

        // Probability calculation.
        let mut results = vec![];

        for (seed, stats) in self.stats.iter().enumerate() {
            let n = if stats.n > 0 { stats.n as f32 } else { 1.0 };
            let name = &sim.teams.names[seed];
            let mean = stats.p_mean;
            let variance = stats.p_ds / n;
            let std_deviation = variance.sqrt();

            results.push((name, mean, std_deviation));
        }

        // Sort by the mean descending.
        results.sort_by_key(|(_, mean, _)| (*mean * -100.0) as i32);

        // Print table header
        out.push(format!(
            "\n{blurb}\n\n{name:<24}{mean:<20}{std:<20}",
            blurb = "Probability of beating opponents faced - mean probability of beating the opponents faced per team.",
            name = "Team",
            mean = "Mean Win %",
            std = "Std. Deviation"
        ));

        // Format each result into a string.
        for (i, (name, mean, std)) in results.into_iter().enumerate() {
            out.push(format!(
                "{num:<4}{name:<20}{mean:<20.3}+/- {std:<16.3}",
                num = format!("{}.", i + 1),
            ));
        }

        // Difficulty calculation.
        let mut results = vec![];

        for (seed, stats) in self.stats.iter().enumerate() {
            let n = if stats.n > 0 { stats.n as f32 } else { 1.0 };
            let name = &sim.teams.names[seed];
            let mean = (stats.r_mean - baseline) / baseline * 100.0;
            let variance = stats.r_ds / n;
            let std_deviation = variance.sqrt() / baseline * 100.0;

            results.push((name, mean, std_deviation));
        }

        // Sort by the mean descending.
        results.sort_by_key(|(_, mean, _)| (*mean * -100.0) as i32);

        // Print table header
        out.push(format!(
            "\n{blurb}\n\n{name:<24}{mean:<20}{std:<20}",
            blurb = "Difficulty of opponents faced - percentage difference between the mean rating of all teams versus the mean rating of opponents faced per team.",
            name = "Team",
            mean = "Mean Difficulty",
            std = "Std. Deviation"
        ));

        // Format each result into a string.
        for (i, (name, mean, std)) in results.into_iter().enumerate() {
            out.push(format!(
                "{num:<4}{name:<20}{mean:<20.3}+/- {std:<16.3}",
                num = format!("{}.", i + 1),
            ));
        }

        out.join("\n")
    }
}
