use rand::prelude::*;

use crate::{
    datatypes::{Index, Rating, Set},
    simulation::{Matchups, calculate_probabilities},
};

/// Mutable state for one Swiss-system tournament iteration.
#[derive(Debug, Clone, Copy)]
pub struct SwissSystem {
    /// Match wins per team.
    pub wins: [u8; 16],
    /// Match losses per team.
    pub losses: [u8; 16],
    /// Win-loss differential per team, used for record-group sorting.
    pub diffs: [i8; 16],
    /// Opponents already faced by each team.
    pub opponents: [Set; 16],
    /// Best-of-one win probability matrix, indexed by `[team_a][team_b]`.
    pub probabilities_bo1: [[f32; 16]; 16],
    /// Best-of-three win probability matrix, indexed by `[team_a][team_b]`.
    pub probabilities_bo3: [[f32; 16]; 16],
    /// Team ratings sorted by initial seed.
    pub ratings: [Rating; 16],
    /// Teams that have not yet advanced or been eliminated.
    pub remaining: Set,
    /// Number of completed tournament rounds.
    pub rounds_complete: u8,
}

impl SwissSystem {
    #[must_use]
    #[cfg_attr(feature = "pprof", inline(never))]
    /// Create a fresh tournament state and precompute matchup probabilities.
    pub fn new(ratings: [Rating; 16], sigma: f32) -> Self {
        let [probabilities_bo1, probabilities_bo3] = calculate_probabilities(ratings, sigma);
        let wins = [0; 16];
        let losses = [0; 16];
        let diffs = [0; 16];
        let opponents = [Set::new(); 16];

        Self {
            wins,
            losses,
            diffs,
            opponents,
            probabilities_bo1,
            probabilities_bo3,
            ratings,
            remaining: Set::full(),
            rounds_complete: 0,
        }
    }

    /// Reset Swiss System state to restart tournament.
    #[cfg_attr(feature = "pprof", inline(never))]
    #[cfg_attr(not(feature = "pprof"), inline)]
    pub const fn reset(&mut self) {
        self.wins = [0; 16];
        self.losses = [0; 16];
        self.diffs = [0; 16];
        self.opponents = [Set::new(); 16];
        self.remaining = Set::full();
        self.rounds_complete = 0;
    }

    /// Simulate one independent match and update records, opponents, and status.
    #[cfg_attr(feature = "pprof", inline(never))]
    fn simulate_match<R: rand::Rng>(&mut self, rng: &mut R, seed_a: Index, seed_b: Index) {
        let r = rng.random();
        let a = seed_a.to_usize();
        let b = seed_b.to_usize();

        // Advancement and elimination matches are BO3; all other matches are BO1.
        let is_bo3 = self.wins[a] == 2 || self.losses[a] == 2;

        // Simulate match outcome.
        let p = if is_bo3 {
            self.probabilities_bo3[a][b]
        } else {
            self.probabilities_bo1[a][b]
        };

        let team_a_win = p > r;

        // Update team records.
        if team_a_win {
            self.wins[a] += 1;
            self.losses[b] += 1;
            self.diffs[a] += 1;
            self.diffs[b] -= 1;
        } else {
            self.losses[a] += 1;
            self.wins[b] += 1;
            self.diffs[a] -= 1;
            self.diffs[b] += 1;
        }

        self.opponents[a].insert(seed_b);
        self.opponents[b].insert(seed_a);

        // A team can only reach three wins or losses in a BO3 round, so status
        // changes are limited to advancement/elimination matches.
        if is_bo3 {
            if self.wins[a] == 3 || self.losses[a] == 3 {
                self.remaining.remove(seed_a);
            }

            if self.wins[b] == 3 || self.losses[b] == 3 {
                self.remaining.remove(seed_b);
            }
        }
    }

    /// Simulate one tournament round.
    #[cfg_attr(feature = "pprof", inline(never))]
    #[cfg_attr(not(feature = "pprof"), inline)]
    fn simulate_round<R: rand::Rng>(&mut self, rng: &mut R) {
        for (a, b) in Matchups::new(self) {
            self.simulate_match(rng, a, b);
        }

        self.rounds_complete += 1;
    }

    /// Simulate all five Swiss rounds.
    #[cfg_attr(feature = "pprof", inline(never))]
    #[cfg_attr(not(feature = "pprof"), inline)]
    pub fn simulate_tournament<R: rand::Rng>(&mut self, rng: &mut R) {
        while self.rounds_complete < 5 {
            self.simulate_round(rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{datatypes::Teams, simulation::rng};

    macro_rules! set {
        ($($n:expr),*) => {
            [$(Index::new::<$n>(),)*].into_iter().collect()
        };
    }

    /// Exact regression test, will break if the seeding algorithm changes.
    /// Uses fake RNG to isolate algorithmic changes from micro statistical changes.
    #[test]
    fn exact_regression_test() {
        let mut ss = SwissSystem::new(Teams::dummy().ratings, 800.0);
        ss.simulate_tournament(&mut rng::HalfRng);

        assert_eq!(ss.wins, [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0]);
        assert_eq!(ss.losses, [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]);
        assert_eq!(
            ss.opponents,
            [
                set!(3, 7, 8),
                set!(2, 6, 9),
                set!(1, 5, 7, 10),
                set!(0, 4, 6, 11),
                set!(3, 5, 11, 12),
                set!(2, 4, 9, 10, 13),
                set!(1, 3, 8, 9, 14),
                set!(0, 2, 8, 10, 15),
                set!(0, 6, 7, 13, 15),
                set!(1, 5, 6, 12, 14),
                set!(2, 5, 7, 11, 13),
                set!(3, 4, 10, 12),
                set!(4, 9, 11, 15),
                set!(5, 8, 10, 14),
                set!(6, 9, 13),
                set!(7, 8, 12),
            ]
        );
    }

    /// Statistical regression test, will break on material distribution changes.
    #[test]
    #[allow(clippy::cast_precision_loss, clippy::unreadable_literal)]
    fn statistical_regression_test() {
        const ITERATIONS: usize = 100_000;
        const ITERATIONS_F32: f32 = ITERATIONS as f32;
        const TOLERANCE: f32 = 0.005;

        let fresh_ss = SwissSystem::new(Teams::dummy().ratings, 800.0);
        let mut rng = rng::deterministic();
        let mut total_three_zero = [0_u64; 16];
        let mut total_advancing = [0_u64; 16];
        let mut total_zero_three = [0_u64; 16];

        for _ in 0..ITERATIONS {
            let mut ss = fresh_ss;
            ss.simulate_tournament(&mut rng);

            for seed in 0..16 {
                let wins = ss.wins[seed];
                let losses = ss.losses[seed];

                if wins == 3 && losses == 0 {
                    total_three_zero[seed] += 1;
                }

                if wins == 3 && losses != 0 {
                    total_advancing[seed] += 1;
                }

                if wins == 0 && losses == 3 {
                    total_zero_three[seed] += 1;
                }
            }
        }

        let expected_three_zero = [
            0.467134, 0.381915, 0.30356, 0.239474, 0.18577, 0.141047, 0.106158, 0.077854, 0.029252,
            0.022126, 0.016032, 0.010871, 0.007562, 0.005146, 0.003569, 0.00253,
        ];

        let expected_advancing = [
            0.482817, 0.542122, 0.585634, 0.604685, 0.604085, 0.584402, 0.547824, 0.497656,
            0.394673, 0.324406, 0.258943, 0.199543, 0.148371, 0.105615, 0.071796, 0.047428,
        ];

        let expected_zero_three = [
            0.002564, 0.003679, 0.005201, 0.007576, 0.010758, 0.01579, 0.021963, 0.029273, 0.07803,
            0.105759, 0.141881, 0.18603, 0.238006, 0.303477, 0.383103, 0.46691,
        ];

        for (actual_counts, expected) in [
            (total_three_zero, expected_three_zero),
            (total_advancing, expected_advancing),
            (total_zero_three, expected_zero_three),
        ] {
            let actual = actual_counts.map(|count| count as f32 / ITERATIONS_F32);

            for seed in 0..16 {
                assert!(
                    (actual[seed] - expected[seed]).abs() < TOLERANCE,
                    "seed {seed}: actual {}, expected {}\n\nActual: {actual:#?}\n\nExpected: {expected:#?}",
                    actual[seed],
                    expected[seed]
                );
            }
        }
    }
}
