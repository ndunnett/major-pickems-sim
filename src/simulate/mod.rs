use std::{path::PathBuf, time::Instant};

use itertools::Itertools;
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{data::parse_toml, simulate::reporting::Report};

mod reporting;
mod swiss_system;
mod team_set;

pub use reporting::{BasicReport, NullReport};
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
    pub fn bench_test<R: Report>(iterations: u64) -> R {
        let sim = Self::dummy(iterations);
        let fresh_ss = SwissSystem::new(sim.ratings, sim.sigma);
        let mut report = R::default();
        let mut rng = make_deterministic_rng();

        for _ in 0..iterations {
            let mut ss = fresh_ss;
            ss.simulate_tournament(&mut rng);
            report.update(&ss);
        }

        report
    }

    /// Run a tournament simulation to completion and return a report.
    pub fn run<R: Report>(&self) -> R {
        let fresh_ss = SwissSystem::new(self.ratings, self.sigma);

        (0..self.iterations)
            .into_par_iter()
            .map_init(
                || (fresh_ss, make_rng()),
                |(ss, rng), _| {
                    ss.reset();
                    ss.simulate_tournament(rng);
                    R::from_swiss_system(ss)
                },
            )
            .sum()
    }
}

/// Run a tournament simulation and print the report.
pub fn simulate<R: Report>(file: PathBuf, sigma: f32, iterations: u64) -> anyhow::Result<()> {
    let now = Instant::now();
    let (names, ratings) = parse_toml(file)?;
    let sim = Simulation::new(names, ratings, sigma, iterations);
    let report = sim.run::<R>();

    println!(
        "{}\n\nRun time: {} seconds",
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
        let report = Simulation::bench_test::<BasicReport>(iterations);

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
                .map(|index| report.stats[index].advanced)
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

    /// Regression test, will break if the seeding algorithm changes.
    #[test]
    fn regression_test() {
        let mut rng = make_deterministic_rng();
        let sim = Simulation::dummy(1);
        let mut ss = SwissSystem::new(sim.ratings, sim.sigma);
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
