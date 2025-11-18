use std::{iter::Sum, ops::Add};

use itertools::Itertools;

use crate::simulate::{Simulation, SwissSystem};

/// Interface for a generic report type to gather information from simulation iterations and formulate a report.
pub trait Report: Copy + Default + Send + Sum {
    fn update(&mut self, ss: &SwissSystem);
    fn format(&self, sim: &Simulation) -> String;

    fn from_swiss_system(ss: &SwissSystem) -> Self {
        let mut report = Self::default();
        report.update(ss);
        report
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BasicStats {
    pub three_zero: u64,
    pub advanced: u64,
    pub zero_three: u64,
}

impl Add for BasicStats {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.three_zero += rhs.three_zero;
        self.advanced += rhs.advanced;
        self.zero_three += rhs.zero_three;
        self
    }
}

/// Report for basic statistic gathering; 3-0, advancment, and 0-3 percentages for each team.
#[derive(Debug, Clone, Copy, Default)]
pub struct BasicReport {
    pub stats: [BasicStats; 16],
}

impl Sum for BasicReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |mut acc, report| {
            for i in 0..acc.stats.len() {
                acc.stats[i] = acc.stats[i] + report.stats[i];
            }

            acc
        })
    }
}

impl Report for BasicReport {
    fn update(&mut self, ss: &SwissSystem) {
        for (seed, result) in self.stats.iter_mut().enumerate() {
            match (ss.wins[seed], ss.losses[seed]) {
                (3, 0) => result.three_zero += 1,
                (3, _) => result.advanced += 1,
                (0, 3) => result.zero_three += 1,
                _ => {}
            }
        }
    }

    fn format(&self, sim: &Simulation) -> String {
        let mut out = Vec::new();

        // Setup access functions and titles for each field of stats.
        type TeamResultField = fn(&BasicStats) -> u64;
        let fields: [(TeamResultField, &str); 3] = [
            (|result| result.three_zero, "3-0"),
            (|result| result.advanced, "3-1 or 3-2"),
            (|result| result.zero_three, "0-3"),
        ];

        // Process each field of stats.
        for (func, title) in fields.iter() {
            out.push(format!("\nMost likely to {title}:"));

            // Sort results from highest to lowest.
            let sorted_results = sim
                .names
                .iter()
                .zip(self.stats.iter())
                .sorted_by(|(_, a), (_, b)| func(b).cmp(&func(a)))
                .enumerate();

            // Format each result into a string.
            for (i, (name, result)) in sorted_results {
                out.push(format!(
                    "{num:<4}{name:<20}{percent:>6.1}%",
                    num = format!("{}.", i + 1),
                    name = name,
                    percent = (func(result) as f32 / sim.iterations as f32 * 1000.0).round() / 10.0
                ));
            }
        }

        out.join("\n")
    }
}

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

impl Sum for StrengthReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |mut acc, report| {
            for (seed, a) in acc.stats.iter_mut().enumerate() {
                let b = report.stats[seed];

                if a.n == 0 {
                    *a = report.stats[seed];
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

            acc
        })
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

                // Update ratings stats
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

/// Report which composes all other reports.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReportAll {
    pub basic: BasicReport,
    pub strength: StrengthReport,
}

impl Sum for ReportAll {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let (basic_iter, strength_iter): (Vec<_>, Vec<_>) =
            iter.map(|report| (report.basic, report.strength)).unzip();

        ReportAll {
            basic: basic_iter.into_iter().sum(),
            strength: strength_iter.into_iter().sum(),
        }
    }
}

impl Report for ReportAll {
    fn update(&mut self, ss: &SwissSystem) {
        self.basic.update(ss);
        self.strength.update(ss);
    }

    fn format(&self, sim: &Simulation) -> String {
        format!("{}\n{}", self.basic.format(sim), self.strength.format(sim))
    }
}

/// Report type to use for benchmarking without optimising away simulation.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullReport(usize);

impl Sum for NullReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        NullReport(std::hint::black_box(iter.count()))
    }
}

impl Report for NullReport {
    fn update(&mut self, ss: &SwissSystem) {
        self.0 = std::hint::black_box(ss.opponents.len());
    }

    fn format(&self, _: &Simulation) -> String {
        String::from("<NullReport>")
    }
}
