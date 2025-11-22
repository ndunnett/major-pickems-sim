use std::{iter::Sum, ops::Add};

use crate::simulate::{Simulation, SwissSystem, reporting::Report};

#[derive(Debug, Clone, Copy, Default)]
pub struct DistributionStats {
    pub r_mean: f32,
    pub r_ds: f32,
    pub p_mean: f32,
    pub p_ds: f32,
    pub n: u64,
}

/// Report to record relative strength of opponents faced for each team.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrengthReport {
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

                // Combine ratings stats
                let r_delta = a.r_mean - b.r_mean;
                a.r_mean = (na * a.r_mean + nb * b.r_mean) / n;
                a.r_ds += b.r_ds + r_delta * r_delta * (na * nb / n);

                // Combine probability stats
                let p_delta = a.p_mean - b.p_mean;
                a.p_mean = (na * a.p_mean + nb * b.p_mean) / n;
                a.p_ds += b.p_ds + p_delta * p_delta * (na * nb / n);

                // Combine count
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
                // Update count
                result.n += 1;

                // Update ratings stats
                let r = ss.ratings[opponent as usize] as f32;
                let r_delta1 = r - result.r_mean;
                result.r_mean += r_delta1 / result.n as f32;
                let r_delta2 = r - result.r_mean;
                result.r_ds += r_delta1 * r_delta2;

                // Update probability stats
                let p = ss.probabilities_bo1[seed][opponent as usize] * 100.0;
                let p_delta1 = p - result.p_mean;
                result.p_mean += p_delta1 / result.n as f32;
                let p_delta2 = p - result.p_mean;
                result.p_ds += p_delta1 * p_delta2;
            }
        }
    }

    fn format(&self, sim: &Simulation) -> String {
        let mut out = Vec::new();

        // Get mean rating of teams
        let baseline = sim.ratings.iter().copied().map(f32::from).sum::<f32>() / 16.0;

        // Probability calculation
        let mut results = vec![];

        for (seed, stats) in self.stats.iter().enumerate() {
            let n = if stats.n > 0 { stats.n as f32 } else { 1.0 };
            let name = &sim.names[seed];
            let mean = stats.p_mean;
            let variance = stats.p_ds / n;
            let std_deviation = variance.sqrt();

            results.push((name, mean, std_deviation));
        }

        // Sort by the mean decending
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

        // Difficulty calculation
        let mut results = vec![];

        for (seed, stats) in self.stats.iter().enumerate() {
            let n = if stats.n > 0 { stats.n as f32 } else { 1.0 };
            let name = &sim.names[seed];
            let mean = (stats.r_mean - baseline) / baseline * 100.0;
            let variance = stats.r_ds / n;
            let std_deviation = variance.sqrt() / baseline * 100.0;

            results.push((name, mean, std_deviation));
        }

        // Sort by the mean decending
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
