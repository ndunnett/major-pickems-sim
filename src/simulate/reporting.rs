use std::{
    collections::{BinaryHeap, HashSet},
    iter::Sum,
    ops::Add,
};

use itertools::Itertools;

use crate::{
    data::TeamSeed,
    simulate::{Simulation, SwissSystem},
};

/// Interface for a generic report type to gather information from simulation iterations and formulate a report.
pub trait Report: Add + Copy + Default + Send + Sum {
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

impl BasicReport {
    fn calculate_probabilities(&self, sim: &Simulation) -> [[f32; 16]; 3] {
        let n = sim.iterations as f32;
        let [mut three_zero, mut advanced, mut zero_three] = [[0.0; 16]; 3];

        for seed in 0..16 {
            three_zero[seed] += self.stats[seed].three_zero as f32;
            advanced[seed] += self.stats[seed].advanced as f32;
            zero_three[seed] += self.stats[seed].zero_three as f32;
        }

        for seed in 0..16 {
            three_zero[seed] /= n;
            advanced[seed] /= n;
            zero_three[seed] /= n;
        }

        [three_zero, advanced, zero_three]
    }
}

impl Add for BasicReport {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..self.stats.len() {
            self.stats[i] = self.stats[i] + rhs.stats[i];
        }

        self
    }
}

impl Sum for BasicReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, report| acc + report)
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
        let probabilities = self.calculate_probabilities(sim);
        let mut out = Vec::new();

        // Setup access indices and titles for each field of stats.
        let fields: [(usize, &str); 3] = [(0, "3-0"), (1, "3-1 or 3-2"), (2, "0-3")];

        // Process each field of stats.
        for (index, title) in fields.into_iter() {
            out.push(format!("\nMost likely to {title}:"));

            // Sort results from highest to lowest.
            let sorted_results = sim
                .names
                .iter()
                .zip(probabilities[index].into_iter())
                .sorted_by(|(_, a), (_, b)| b.total_cmp(a))
                .enumerate();

            // Format each result into a string.
            for (i, (name, result)) in sorted_results {
                out.push(format!(
                    "{num:<4}{name:<20}{percent:>6.1}%",
                    num = format!("{}.", i + 1),
                    name = name,
                    percent = (result * 1000.0).round() / 10.0
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

/// Report for selecting optimal picks from basic statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct PicksReport {
    pub basic: BasicReport,
}

impl Add for PicksReport {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.basic = self.basic + rhs.basic;
        self
    }
}

impl Sum for PicksReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, report| acc + report)
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    seed: TeamSeed,
    probability: f32,
}

impl std::hash::Hash for Candidate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.seed.hash(state);
    }
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.probability.total_cmp(&other.probability)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Report for PicksReport {
    fn update(&mut self, ss: &SwissSystem) {
        self.basic.update(ss);
    }

    fn format(&self, sim: &Simulation) -> String {
        let [three_zero, advanced, zero_three] = self.basic.calculate_probabilities(sim);

        // Sort out candidates for each category of picks.
        let candidates = |probabilities: [f32; 16]| -> BinaryHeap<Candidate> {
            probabilities
                .iter()
                .enumerate()
                .map(|(i, p)| Candidate {
                    seed: i as TeamSeed,
                    probability: *p,
                })
                .collect::<BinaryHeap<_>>()
        };

        let mut three_zero_candidates = candidates(three_zero);
        let mut advanced_candidates = candidates(advanced);
        let mut zero_three_candidates = candidates(zero_three);

        // Naively select 3-1/3-2 picks.
        let mut advanced_picks = HashSet::new();

        for _ in 0..6 {
            if let Some(candidate) = advanced_candidates.pop() {
                advanced_picks.insert(candidate);
            }
        }

        // Naively select 0-3 picks.
        let mut zero_three_picks = HashSet::new();

        for _ in 0..2 {
            if let Some(candidate) = zero_three_candidates.pop() {
                zero_three_picks.insert(candidate);
            }
        }

        // Optimise 3-0 picks by swapping previous 3-1/3-2 picks to maximise win probability.
        let mut three_zero_picks = HashSet::new();

        while three_zero_picks.len() < 2 {
            let mut swap_candidates = BinaryHeap::new();

            while let Some(candidate) = three_zero_candidates.peek()
                && advanced_picks.contains(candidate)
            {
                swap_candidates.push(three_zero_candidates.pop().unwrap());
            }

            match (
                three_zero_candidates.pop(),
                swap_candidates.pop(),
                advanced_candidates.pop(),
            ) {
                // There are teams in all relevant candidate pools.
                (Some(next_three_zero), Some(next_swap), Some(next_advanced))
                    if advanced_picks.contains(&next_swap) =>
                {
                    let swap_advanced = advanced_picks.get(&next_swap).unwrap();

                    // Calculate delta in win probability for swapping teams.
                    let cost = next_advanced.probability - swap_advanced.probability;
                    let reward = next_swap.probability - next_three_zero.probability;

                    // Swap teams if it is worthwhile, and repopulate candidate pools with candidates that remain unselected.
                    if reward > cost {
                        three_zero_picks.insert(next_swap);
                        advanced_picks.remove(&next_swap);
                        advanced_picks.insert(next_advanced);
                        three_zero_candidates.push(next_three_zero);
                    } else {
                        three_zero_picks.insert(next_three_zero);
                        advanced_candidates.push(next_advanced);
                    }
                }
                // There are only teams left in the 3-0 candidate pool, and unviable candidates in the 3-1/3-2 pool.
                (Some(next_three_zero), None, Some(next_advanced)) => {
                    three_zero_picks.insert(next_three_zero);
                    advanced_candidates.push(next_advanced);
                }
                // There are only teams left in the 3-0 candidate pool.
                (Some(next_three_zero), None, None) => {
                    three_zero_picks.insert(next_three_zero);
                }
                // The current state no longer makes any sense, either the 3-0 pool is empty or the 3-1/3-2 picks don't contain the next swap candidate.
                state => {
                    unreachable!("invalid state for picking 3-0 teams:\n\n{state:#?}");
                }
            }
        }

        // Estimate how many stars will be earned on average for the given picks.
        let estimated_stars = three_zero_picks
            .iter()
            .chain(advanced_picks.iter())
            .chain(zero_three_picks.iter())
            .map(|candidate| candidate.probability)
            .sum::<f32>();

        let format_picks = |out: &mut Vec<String>, picks: &HashSet<Candidate>| {
            for (i, (name, p)) in picks
                .iter()
                .map(|i| (&sim.names[i.seed as usize], i.probability * 100.0))
                .sorted_by(|(_, a), (_, b)| b.total_cmp(a))
                .enumerate()
            {
                out.push(format!(
                    "{num:<4}{name:<20}{p:>6.1}%",
                    num = format!("{}.", i + 1),
                ));
            }
        };

        let mut out = vec![String::from("\n3-0 picks:")];
        format_picks(&mut out, &three_zero_picks);
        out.push(String::from("\n3-1 or 3-2 picks:"));
        format_picks(&mut out, &advanced_picks);
        out.push(String::from("\n0-3 picks:"));
        format_picks(&mut out, &zero_three_picks);
        out.push(format!("\nEstimated points: {estimated_stars:.3} / 8"));
        out.join("\n")
    }
}

/// Report which composes all other reports.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReportAll {
    pub basic: BasicReport,
    pub strength: StrengthReport,
    pub picks: PicksReport,
}

impl Add for ReportAll {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.basic = self.basic + rhs.basic;
        self.strength = self.strength + rhs.strength;
        self.picks = self.picks + rhs.picks;
        self
    }
}

impl Sum for ReportAll {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, report| acc + report)
    }
}

impl Report for ReportAll {
    fn update(&mut self, ss: &SwissSystem) {
        self.basic.update(ss);
        self.strength.update(ss);
        self.picks.update(ss);
    }

    fn format(&self, sim: &Simulation) -> String {
        format!(
            "{}\n{}\n{}",
            self.picks.format(sim),
            self.basic.format(sim),
            self.strength.format(sim)
        )
    }
}

/// Report type to use for benchmarking without optimising away simulation.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullReport(usize);

impl Add for NullReport {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.0 = std::hint::black_box(rhs.0);
        self
    }
}

impl Sum for NullReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, report| acc + report)
    }
}

impl Report for NullReport {
    fn update(&mut self, _ss: &SwissSystem) {
        self.0 = std::hint::black_box(0);
    }

    fn format(&self, _: &Simulation) -> String {
        String::from("<NullReport>")
    }
}
