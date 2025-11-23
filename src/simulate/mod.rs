use std::{path::PathBuf, time::Instant};

use itertools::Itertools;
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::data::parse_toml;

mod matching;
mod reporting;
mod swiss_system;
mod team_set;

use matching::MatchupGenerator;
pub use reporting::*;
use swiss_system::SwissSystem;
use team_set::TeamSet;

pub(super) type RngType = rand_chacha::ChaCha8Rng;

/// Random number generator for normal use.
pub(super) fn make_rng() -> RngType {
    RngType::from_rng(&mut rand::rng())
}

/// Deterministic random number generator for testing/benchmarking.
pub(super) fn make_deterministic_rng() -> RngType {
    RngType::seed_from_u64(7355608)
}

/// Instance of a simulation, to parse team data and run tournament iterations.
#[derive(Debug, Clone)]
pub struct Simulation {
    pub names: [String; 16],
    pub ratings: [i16; 16],
    pub sigma: f32,
    pub iterations: u64,
}

impl Simulation {
    pub fn new(names: [String; 16], ratings: [i16; 16], sigma: f32, iterations: u64) -> Self {
        Self {
            names,
            ratings,
            sigma,
            iterations,
        }
    }

    /// Produce simulation with dummy data for testing purposes.
    pub fn dummy(iterations: u64) -> Self {
        Self {
            names: (0..16)
                .map(|i| format!("Team {}", i + 1))
                .collect_array()
                .unwrap(),
            ratings: (0..16).map(|i| 2000 - 50 * i).collect_array().unwrap(),
            sigma: 800.0,
            iterations,
        }
    }

    /// Run single-threaded bench test for profiling/benchmarking purposes.
    pub fn bench_test<R: Report>(iterations: u64, mut report: R) -> R {
        let sim = Self::dummy(iterations);
        let fresh_ss = SwissSystem::new(sim.ratings, sim.sigma);
        let mut rng = make_deterministic_rng();

        for _ in 0..iterations {
            let mut ss = fresh_ss;
            ss.simulate_tournament(&mut rng);
            report.update(&ss);
        }

        report
    }

    /// Run a tournament simulation to completion and return a report.
    pub fn run<R: Report>(&self, fresh_report: R) -> R {
        let fresh_ss = SwissSystem::new(self.ratings, self.sigma);

        (0..self.iterations)
            .into_par_iter()
            .map_init(
                || (fresh_ss, make_rng()),
                |(ss, rng), _| {
                    ss.reset();
                    ss.simulate_tournament(rng);
                    let mut report = fresh_report;
                    report.update(ss);
                    report
                },
            )
            .sum()
    }
}

/// Run a tournament simulation and print the report.
pub fn simulate<R: Report>(
    file: PathBuf,
    sigma: f32,
    iterations: u64,
    report: R,
) -> anyhow::Result<()> {
    let now = Instant::now();
    let (names, ratings) = parse_toml(file)?;
    let sim = Simulation::new(names, ratings, sigma, iterations);
    let report = sim.run(report);

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

    println!(
        "RESULTS FROM {formatted_iterations} TOURNAMENT SIMULATIONS\n{}\n\nRun time: {} seconds",
        report.format(&sim),
        now.elapsed().as_millis() as f32 / 1000.0
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
        let report = Simulation::bench_test(iterations, BasicReport::default());

        // Total 3-0 stats should sum to 2 per iteration
        assert_eq!(
            (0..16)
                .map(|index| report.stats[index].three_zero)
                .sum::<u64>(),
            iterations * 2
        );

        // Total 3-1/3-2 stats should sum to 6 per iteration
        assert_eq!(
            (0..16)
                .map(|index| report.stats[index].advancing)
                .sum::<u64>(),
            iterations * 6
        );

        // Total 0-3 stats should sum to 2 per iteration
        assert_eq!(
            (0..16)
                .map(|index| report.stats[index].zero_three)
                .sum::<u64>(),
            iterations * 2
        );

        // Best team should always have more 3-0 stats than the worst team
        assert!(report.stats[0].three_zero > report.stats[15].three_zero);

        // Best team should always have less 0-3 stats than the worst team
        assert!(report.stats[0].zero_three < report.stats[15].zero_three);
    }
}
