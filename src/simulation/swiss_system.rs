use std::{ops::Neg, simd::StdFloat};

use arrayvec::ArrayVec;
use rand::prelude::*;
use std::simd::prelude::*;

use crate::{
    datatypes::{Index, Rating, Set},
    simulation::{MatchupGenerator, RngType},
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
    // Lane values mirror zero-based seed indices.
    const SEED_LANES: Simd<u16, 16> = {
        let mut seeds = [0; 16];
        let mut i = 1;

        while i < 16 {
            seeds[i] = i as u16;
            i += 1;
        }

        Simd::from_array(seeds)
    };

    #[allow(clippy::many_single_char_names)]
    #[must_use]
    #[cfg_attr(feature = "pprof", inline(never))]
    /// Create a fresh tournament state and precompute matchup probabilities.
    pub fn new(ratings: [Rating; 16], sigma: f32) -> Self {
        const ONE: Simd<f32, 16> = Simd::splat(1.0);
        const TWO: Simd<f32, 16> = Simd::splat(2.0);
        let mut r = [0.0_f32; 16];

        for i in 0..16 {
            r[i] = ratings[i].to_f32();
        }

        // Precalculate independent map win probabilities for every possible
        // matchup. Each row fixes team A and compares it against all team B
        // ratings in SIMD lanes.
        //
        // let Ra = team A rating,  Rb = team B rating,  P = team A win probablity
        // P(Ra, Rb) = 1 / (1 + 10^((Rb - Ra) / sigma))
        //
        // `powf` in SIMD compatible operations: x^y => exp(ln(x) * y)
        //
        // P(Ra, Rb) = recip(1 + exp(ln(10) * (Rb - Ra) / sigma))
        //           = recip(1 + exp(u * (Rb - Ra))),  where u = ln(10) / sigma
        let u = Simd::splat(10.0_f32.ln()) / Simd::splat(sigma);
        let rb = Simd::from_array(r);
        let mut probabilities_bo1 = [[0.0; 16]; 16];

        for i in 0..16 {
            let ra = Simd::splat(r[i]);
            probabilities_bo1[i] = (ONE + (u * (rb - ra)).exp()).recip().to_array();
        }

        // Precalculate best-of-three series win probabilities from the map
        // probabilities. A team wins the series by WW, WLW, or LWW.
        //
        // let Q = series win probability,  P = map win probability
        // Q(W) = P
        // Q(L) = 1 - P
        // Q(WW-) = P * P
        // Q(WLW) = Q(LWW) = P * P * (1 - P)
        //
        // let a = P * P,  b = 1 - P
        // Q = Q(WLW) + Q(LWW) + Q(WW-)
        //   = P * P * (1 - P) + P * P * (1 - P) + P * P
        //   = 2 * a * b + a
        let mut probabilities_bo3 = [[0.0; 16]; 16];

        for i in 0..16 {
            let p = Simd::from_array(probabilities_bo1[i]);
            let a = p * p;
            let b = ONE - p;
            probabilities_bo3[i] = TWO.mul_add(a * b, a).to_array();
        }

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

    /// Return the Buchholz difficulty score for a given team.
    #[cfg_attr(feature = "pprof", inline(never))]
    fn buchholz(&self, team: Index) -> i8 {
        const ONE: Simd<u16, 16> = Simd::splat(1);

        let mask = {
            // Shift the opponent bitset by the lane index so each lane's low
            // bit says whether that seed has been played. Negating 0/1 gives
            // 0 or -1, which can be used as an all-bits mask for `diffs`.
            let shifted = self.opponents[team.to_usize()].splat() >> Self::SEED_LANES;
            (shifted & ONE).cast::<i8>().neg()
        };

        (Simd::from_array(self.diffs) & mask).reduce_sum()
    }

    /// Return remaining team indices sorted by mid-stage seed calculation.
    ///
    /// 1. Current win-loss record
    /// 2. Buchholz difficulty score (sum of win-loss record for each opponent faced)
    /// 3. Initial seeding
    ///
    /// [Rules and Regs - Mid-stage Seed Calculation](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#Mid-Stage-Seed-Calculation)
    #[cfg_attr(feature = "pprof", inline(never))]
    pub(super) fn seed_teams(&self) -> ArrayVec<Index, 16> {
        // Bit-pack seeding information into a 16-bit unsigned integer so one
        // unstable integer sort applies every tiebreak in priority order:
        // [15] [14 13 12 11 10] [9 8 7 6 5] [4 3 2 1 0]
        //  --   --------------   ---------   ----------
        //   |          |             |            |
        // Spare bit    |    Buchholz difficulty   |
        //          Win-loss                 Initial seed
        const FIFTEEN: Simd<i8, 16> = Simd::splat(15);
        let buchholz_array =
            std::array::from_fn(|i| self.buchholz(unsafe { Index::from_usize(i) }));
        let buchholz = (FIFTEEN - Simd::from_array(buchholz_array)).cast::<u16>();
        let diffs = (FIFTEEN - Simd::from_array(self.diffs)).cast::<u16>();
        let packed = (diffs << 10 | buchholz << 5 | Self::SEED_LANES).to_array();

        // Select only teams that remain in the tournament; teams already at
        // three wins or three losses are no longer paired.
        let mut seeding = ArrayVec::<u16, 16>::new();

        for seed in self.remaining.iter() {
            seeding.push(packed[seed.to_usize()]);
        }

        seeding.sort_unstable();

        // Strip back down to just the zero-based initial seed.
        for seed in &mut seeding {
            *seed &= 0x1F;
        }

        // `Index` is a transparent newtype of `u16`; masking with 0x1F leaves
        // only original seed lanes, which are known to be in 0..16.
        unsafe { std::mem::transmute(seeding) }
    }

    /// Simulate one independent match and update records, opponents, and status.
    #[cfg_attr(feature = "pprof", inline(never))]
    fn simulate_match(&mut self, rng: &mut RngType, seed_a: Index, seed_b: Index) {
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
    fn simulate_round(&mut self, rng: &mut RngType) {
        for (a, b) in MatchupGenerator::new(&*self) {
            self.simulate_match(rng, a, b);
        }

        self.rounds_complete += 1;
    }

    /// Simulate all five Swiss rounds.
    #[cfg_attr(feature = "pprof", inline(never))]
    #[cfg_attr(not(feature = "pprof"), inline)]
    pub fn simulate_tournament(&mut self, rng: &mut RngType) {
        while self.rounds_complete < 5 {
            self.simulate_round(rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{Simulation, make_deterministic_rng};

    macro_rules! i {
        ($n:expr) => {
            Index::new::<$n>()
        };
    }

    /// Regression test, will break if the seeding algorithm changes.
    #[test]
    fn regression_test() {
        let sim = Simulation::dummy(1);
        let mut ss = SwissSystem::new(sim.teams.ratings, sim.sigma);
        ss.simulate_tournament(&mut make_deterministic_rng());

        assert_eq!(ss.wins, [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 1, 0, 0, 1, 1]);
        assert_eq!(ss.losses, [0, 2, 1, 1, 0, 2, 1, 3, 3, 3, 2, 3, 3, 3, 3, 3]);
        assert_eq!(
            ss.opponents,
            [
                [i!(6), i!(7), i!(8)].into_iter().collect(),
                [i!(2), i!(6), i!(8), i!(9), i!(11)].into_iter().collect(),
                [i!(1), i!(4), i!(5), i!(10)].into_iter().collect(),
                [i!(4), i!(7), i!(9), i!(11)].into_iter().collect(),
                [i!(2), i!(3), i!(12)].into_iter().collect(),
                [i!(2), i!(7), i!(10), i!(11), i!(13)].into_iter().collect(),
                [i!(0), i!(1), i!(10), i!(14)].into_iter().collect(),
                [i!(0), i!(3), i!(5), i!(8), i!(15)].into_iter().collect(),
                [i!(0), i!(1), i!(7), i!(14), i!(15)].into_iter().collect(),
                [i!(1), i!(3), i!(10), i!(14), i!(15)].into_iter().collect(),
                [i!(2), i!(5), i!(6), i!(9), i!(13)].into_iter().collect(),
                [i!(1), i!(3), i!(5), i!(12)].into_iter().collect(),
                [i!(4), i!(11), i!(15)].into_iter().collect(),
                [i!(5), i!(10), i!(14)].into_iter().collect(),
                [i!(6), i!(8), i!(9), i!(13)].into_iter().collect(),
                [i!(7), i!(8), i!(9), i!(12)].into_iter().collect(),
            ]
        );
    }
}
