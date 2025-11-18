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

impl BasicStats {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            three_zero: 0,
            advanced: 0,
            zero_three: 0,
        }
    }
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
        // Format number of iterations into a string, with thousands separated by commas.
        let formatted_iterations = sim
            .iterations
            .to_string()
            .as_bytes()
            .rchunks(3)
            .rev()
            .map(str::from_utf8)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join(",");

        let mut out = vec![format!(
            "RESULTS FROM {formatted_iterations} TOURNAMENT SIMULATIONS"
        )];

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
