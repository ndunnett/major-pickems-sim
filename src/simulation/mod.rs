use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{datatypes::Teams, reporting::Report};

mod matching;
mod swiss_system;

use matching::MatchupGenerator;
pub use swiss_system::SwissSystem;

pub type RngType = rand_chacha::ChaCha8Rng;

/// Random number generator for normal use.
#[must_use]
pub fn make_rng() -> RngType {
    RngType::from_rng(&mut rand::rng())
}

/// Deterministic random number generator for testing/benchmarking.
#[must_use]
pub fn make_deterministic_rng() -> RngType {
    RngType::seed_from_u64(7_355_608)
}

/// Instance of a simulation, to parse team data and run tournament iterations.
#[derive(Debug, Clone)]
pub struct Simulation {
    pub teams: Teams,
    pub sigma: f32,
    pub iterations: u64,
}

impl Simulation {
    #[must_use]
    pub const fn new(teams: Teams, sigma: f32, iterations: u64) -> Self {
        Self {
            teams,
            sigma,
            iterations,
        }
    }

    /// Produce simulation with dummy data for testing purposes.
    #[must_use]
    pub fn dummy(iterations: u64) -> Self {
        Self {
            teams: Teams::dummy(),
            sigma: 800.0,
            iterations,
        }
    }

    /// Run single-threaded bench test for profiling/benchmarking purposes.
    pub fn bench_test<R: Report>(iterations: u64, mut report: R) -> R {
        let sim = Self::dummy(iterations);
        let fresh_ss = SwissSystem::new(sim.teams.ratings, sim.sigma);
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
        let fresh_ss = SwissSystem::new(self.teams.ratings, self.sigma);

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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::reporting::BasicReport;

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
