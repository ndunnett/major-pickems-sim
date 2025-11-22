use std::{
    iter::Sum,
    ops::{Add, AddAssign},
};

use itertools::Itertools;

use crate::simulate::{Simulation, SwissSystem, reporting::Report};

#[derive(Debug, Clone, Copy, Default)]
pub struct BasicStats {
    pub three_zero: u64,
    pub advanced: u64,
    pub zero_three: u64,
}

impl AddAssign for BasicStats {
    fn add_assign(&mut self, rhs: Self) {
        self.three_zero += rhs.three_zero;
        self.advanced += rhs.advanced;
        self.zero_three += rhs.zero_three;
    }
}

/// Report for basic statistic gathering; 3-0, advancment, and 0-3 percentages for each team.
#[derive(Debug, Clone, Copy, Default)]
pub struct BasicReport {
    pub stats: [BasicStats; 16],
}

impl BasicReport {
    pub(super) fn calculate_probabilities(&self, sim: &Simulation) -> [[f32; 16]; 3] {
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
            self.stats[i] += rhs.stats[i];
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
