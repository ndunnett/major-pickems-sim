use std::num::NonZero;

use crate::{
    datatypes::{Iterations, Sigma, Teams},
    reporting::Report,
};

mod matching;
pub mod probabilities;
mod rng;
pub mod seeding;
pub mod sorting;
mod swiss_system;

pub use matching::Matchups;
pub use probabilities::calculate_probabilities;
pub use seeding::seed_teams;
pub use swiss_system::SwissSystem;

/// Configuration for running repeated tournament simulations.
#[derive(Debug, Clone)]
pub struct Simulation {
    /// Seed-ordered team data.
    pub teams: Teams,
    /// Standard deviation parameter for the logistic win-probability model.
    pub sigma: Sigma,
    /// Number of independent tournaments to simulate.
    pub iterations: Iterations,
}

impl Simulation {
    /// Construct a simulation from team data, sigma, and iteration count.
    #[must_use]
    pub const fn new(teams: Teams, sigma: Sigma, iterations: Iterations) -> Self {
        Self {
            teams,
            sigma,
            iterations,
        }
    }

    /// Produce simulation with dummy data for testing purposes.
    #[must_use]
    pub fn dummy(iterations: Iterations) -> Self {
        Self {
            teams: Teams::dummy(),
            sigma: Sigma::default(),
            iterations,
        }
    }

    /// Run single-threaded bench test for profiling/benchmarking purposes.
    pub fn bench_test<R: Report>(&self, mut report: R) -> R {
        let mut ss = SwissSystem::new(self.teams.ratings, self.sigma);
        let mut rng = rng::deterministic();

        for _ in self.iterations {
            ss.reset();
            ss.simulate_tournament(&mut rng);
            report.update(&ss);
        }

        report
    }

    /// Run a tournament simulation to completion and return a report.
    pub fn run<R: Report>(&self, fresh_report: R) -> R {
        const MIN_ITERS_PER_THREAD: u64 = 15_000;
        let threads = std::thread::available_parallelism().map_or(1, NonZero::get) as u64;
        let iterations = self.iterations.to_u64();
        let max_workers = iterations.div_ceil(MIN_ITERS_PER_THREAD);
        let worker_count = threads.min(max_workers);
        let dividend = iterations / worker_count;
        let remainder = iterations % worker_count;

        std::thread::scope(|scope| {
            let fresh_ss = SwissSystem::new(self.teams.ratings, self.sigma);

            let handles = (0..worker_count)
                .map(|worker| {
                    let worker_iterations = dividend + u64::from(worker < remainder);

                    scope.spawn(move || {
                        let mut ss = fresh_ss;
                        let mut rng = rng::random();
                        let mut report = fresh_report;

                        for _ in 0..worker_iterations {
                            ss.reset();
                            ss.simulate_tournament(&mut rng);
                            report.update(&ss);
                        }

                        report
                    })
                })
                .collect::<Vec<_>>();

            handles
                .into_iter()
                .map(|handle| handle.join().expect("simulation worker panicked"))
                .sum()
        })
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
        let report =
            Simulation::dummy(Iterations::new(iterations)).bench_test(BasicReport::default());

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
