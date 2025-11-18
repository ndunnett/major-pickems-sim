use std::{
    ops::Add,
    path::PathBuf,
    time::{Duration, Instant},
};

use itertools::Itertools;
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::data::parse_toml;

mod swiss_system;
mod team_set;

use swiss_system::SwissSystem;
use team_set::TeamSet;

type RngType = rand_chacha::ChaCha8Rng;

/// Random number generator for normal use.
fn make_rng() -> RngType {
    RngType::from_rng(&mut rand::rng())
}

/// Deterministic random number generator for testing/benchmarking.
fn make_deterministic_rng() -> RngType {
    RngType::seed_from_u64(7355608)
}

/// Struct to tally up tournament results for a team.
#[derive(Debug, Clone, Copy)]
pub struct TeamResult {
    pub three_zero: u64,
    pub advanced: u64,
    pub zero_three: u64,
}

impl TeamResult {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            three_zero: 0,
            advanced: 0,
            zero_three: 0,
        }
    }
}

impl Default for TeamResult {
    fn default() -> Self {
        Self::new()
    }
}

impl Add for TeamResult {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.three_zero += rhs.three_zero;
        self.advanced += rhs.advanced;
        self.zero_three += rhs.zero_three;
        self
    }
}

type TeamResultField = fn(&TeamResult) -> u64;

/// Instance of a simulation, to parse team data and run tournament iterations.
#[derive(Debug, Clone)]
pub struct Simulation {
    names: [String; 16],
    ratings: [i16; 16],
}

impl Simulation {
    pub fn try_from_file(filepath: PathBuf) -> anyhow::Result<Self> {
        let (names, ratings) = parse_toml(filepath)?;
        Ok(Self { names, ratings })
    }

    /// Produce simulation with dummy data for testing purposes.
    pub fn dummy() -> Self {
        Self {
            names: (0..16)
                .map(|i| format!("Team {}", i + 1))
                .collect_array()
                .unwrap(),
            ratings: (0..16).map(|i| 2000 - 50 * i).collect_array().unwrap(),
        }
    }

    /// Run single-threaded bench test for profiling/benchmarking purposes.
    pub fn bench_test(iterations: usize) -> [TeamResult; 16] {
        let sim = Self::dummy();
        let mut results = [TeamResult::new(); 16];
        let fresh_ss = SwissSystem::new(sim.ratings, 800.0);
        let mut rng = make_deterministic_rng();

        for _ in 0..iterations {
            let mut ss = fresh_ss;
            ss.simulate_tournament(&mut rng);

            for (seed, result) in results.iter_mut().enumerate() {
                match (ss.wins[seed], ss.losses[seed]) {
                    (3, 0) => result.three_zero += 1,
                    (3, 1 | 2) => result.advanced += 1,
                    (0, 3) => result.zero_three += 1,
                    _ => {}
                }
            }
        }

        results
    }

    /// Run 'n' iterations of tournament simulation.
    pub fn run(&self, n: u64, sigma: f32) -> [TeamResult; 16] {
        let fresh_ss = SwissSystem::new(self.ratings, sigma);
        let fresh_results = [TeamResult::new(); 16];

        (0..n)
            .into_par_iter()
            .map_init(
                || (fresh_ss, make_rng()),
                |(ss, rng), _| {
                    let mut results = fresh_results;
                    ss.reset();
                    ss.simulate_tournament(rng);

                    for (seed, result) in results.iter_mut().enumerate() {
                        match (ss.wins[seed], ss.losses[seed]) {
                            (3, 0) => result.three_zero += 1,
                            (3, 1 | 2) => result.advanced += 1,
                            (0, 3) => result.zero_three += 1,
                            _ => {}
                        }
                    }

                    results
                },
            )
            .reduce(
                || fresh_results,
                |acc, result| {
                    acc.into_iter()
                        .zip(result)
                        .map(|(a, b)| a + b)
                        .collect_array()
                        .unwrap()
                },
            )
    }

    /// Format results from a simulation into a readable/printable string.
    pub fn format_results(
        &self,
        results: &[TeamResult],
        iterations: u64,
        run_time: Duration,
    ) -> String {
        // Format number of iterations into a string, with thousands separated by commas.
        let formatted_iterations = iterations
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
        let fields: [(TeamResultField, &str); 3] = [
            (|result| result.three_zero, "3-0"),
            (|result| result.advanced, "3-1 or 3-2"),
            (|result| result.zero_three, "0-3"),
        ];

        // Process each field of stats.
        for (func, title) in fields.iter() {
            out.push(format!("\nMost likely to {title}:"));

            // Sort results from highest to lowest.
            let sorted_results = self
                .names
                .iter()
                .zip(results.iter())
                .sorted_by(|(_, a), (_, b)| func(b).cmp(&func(a)))
                .enumerate();

            // Format each result into a string.
            for (i, (name, result)) in sorted_results {
                out.push(format!(
                    "{num:<4}{name:<20}{percent:>6.1}%",
                    num = format!("{}.", i + 1),
                    name = name,
                    percent = (func(result) as f32 / iterations as f32 * 1000.0).round() / 10.0
                ));
            }
        }

        out.push(format!(
            "\nRun time: {} seconds",
            run_time.as_millis() as f32 / 1000.0
        ));

        out.join("\n")
    }
}

/// Run simulation and print results.
pub fn simulate(file: PathBuf, iterations: u64, sigma: f32) -> anyhow::Result<()> {
    let now = Instant::now();
    let sim = Simulation::try_from_file(file)?;
    let results = sim.run(iterations, sigma);
    println!(
        "{}",
        sim.format_results(&results, iterations, now.elapsed())
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quick sanity test to check that things are generally working.
    #[test]
    fn sanity_test() {
        let iterations = 1000;
        let results = Simulation::bench_test(iterations);

        // Total 3-0 stats should sum to 2 per iteration
        assert_eq!(
            (0..16_usize)
                .map(|index| results[index].three_zero as usize)
                .sum::<usize>(),
            iterations * 2
        );

        // Total 3-1/3-2 stats should sum to 6 per iteration
        assert_eq!(
            (0..16_usize)
                .map(|index| results[index].advanced as usize)
                .sum::<usize>(),
            iterations * 6
        );

        // Total 0-3 stats should sum to 2 per iteration
        assert_eq!(
            (0..16_usize)
                .map(|index| results[index].zero_three as usize)
                .sum::<usize>(),
            iterations * 2
        );

        // Best team should always have more 3-0 stats than the worst team
        assert!(results[0].three_zero > results[15].three_zero);

        // Best team should always have less 0-3 stats than the worst team
        assert!(results[0].zero_three < results[15].zero_three);
    }

    /// Regression test, will break if the seeding algorithm changes.
    #[test]
    fn regression_test() {
        let mut rng = make_deterministic_rng();
        let sim = Simulation::dummy();
        let mut ss = SwissSystem::new(sim.ratings, 800.0);
        ss.simulate_tournament(&mut rng);

        assert_eq!(ss.wins, [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 1, 0, 0, 1, 1]);
        assert_eq!(ss.losses, [0, 2, 1, 1, 0, 2, 1, 3, 3, 3, 2, 3, 3, 3, 3, 3]);
        assert_eq!(
            ss.opponents,
            [
                TeamSet::from([6, 7, 8]),
                TeamSet::from([2, 6, 8, 9, 11]),
                TeamSet::from([1, 4, 5, 10]),
                TeamSet::from([4, 7, 9, 11]),
                TeamSet::from([2, 3, 12]),
                TeamSet::from([2, 7, 10, 11, 13]),
                TeamSet::from([0, 1, 10, 14]),
                TeamSet::from([0, 3, 5, 8, 15]),
                TeamSet::from([0, 1, 7, 14, 15]),
                TeamSet::from([1, 3, 10, 14, 15]),
                TeamSet::from([2, 5, 6, 9, 13]),
                TeamSet::from([1, 3, 5, 12]),
                TeamSet::from([4, 11, 15]),
                TeamSet::from([5, 10, 14]),
                TeamSet::from([6, 8, 9, 13]),
                TeamSet::from([7, 8, 9, 12]),
            ]
        );
    }
}
