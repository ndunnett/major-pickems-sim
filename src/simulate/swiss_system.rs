use std::{ops::Neg, simd::StdFloat};

use arrayvec::ArrayVec;
use rand::prelude::*;
use std::simd::prelude::*;

use crate::{
    data::TeamSeed,
    simulate::{MatchupGenerator, RngType, TeamSet},
};

/// Instance of a single swiss system iteration.
#[derive(Debug, Clone, Copy)]
pub struct SwissSystem {
    pub(super) wins: [u8; 16],
    pub(super) losses: [u8; 16],
    pub(super) diffs: [i8; 16],
    pub(super) opponents: [TeamSet; 16],
    pub(super) probabilities_bo1: [[f32; 16]; 16],
    pub(super) probabilities_bo3: [[f32; 16]; 16],
    pub(super) ratings: [i16; 16],
    pub(super) remaining: TeamSet,
    pub(super) rounds_complete: u8,
}

impl SwissSystem {
    const SEED_LANES: Simd<u16, 16> = {
        let mut seeds = [0; 16];
        let mut i = 1;

        while i < 16 {
            seeds[i] = i as u16;
            i += 1;
        }

        Simd::from_array(seeds)
    };

    pub fn new(ratings: [i16; 16], sigma: f32) -> Self {
        const ONE: Simd<f32, 16> = Simd::splat(1.0);
        const TWO: Simd<f32, 16> = Simd::splat(2.0);
        let mut r = [0.0_f32; 16];

        for i in 0..16 {
            r[i] = ratings[i] as f32;
        }

        // Precalculate matrix of independent map win probabilities for all possible matchups using SIMD.
        //
        // let Ra = team A rating,  Rb = team B rating,  P = team A win probablity
        // P(Ra, Rb) = 1 / (1 + 10^((Rb - Ra) / sigma))
        // `powf` in SIMD compatible operations: x^y => exp(ln(x) * y)
        // P(Ra, Rb) = recip(1 + exp(ln(10) * (Rb - Ra) / sigma))
        //           = recip(1 + exp(u * (Rb - Ra))),  where u = ln(10) / sigma
        let u = Simd::splat(10.0_f32.ln()) / Simd::splat(sigma);
        let rb = Simd::from_array(r);
        let mut probabilities_bo1 = [[0.0; 16]; 16];

        for i in 0..16 {
            let ra = Simd::splat(r[i]);
            probabilities_bo1[i] = (ONE + (u * (rb - ra)).exp()).recip().to_array();
        }

        // Precalculate matrix of BO3 series win probabilities for all possible matchups using SIMD.
        //
        // let Q = series win probability,  P = map win probability
        // Q(W) = P,  Q(L) = 1 - P
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
        let opponents = [TeamSet::new(); 16];

        Self {
            wins,
            losses,
            diffs,
            opponents,
            probabilities_bo1,
            probabilities_bo3,
            ratings,
            remaining: TeamSet::full(),
            rounds_complete: 0,
        }
    }

    /// Reset Swiss System state to restart tournament.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.wins = [0; 16];
        self.losses = [0; 16];
        self.diffs = [0; 16];
        self.opponents = [TeamSet::new(); 16];
        self.remaining = TeamSet::full();
        self.rounds_complete = 0;
    }

    /// Return the Buchholz difficulty score for a given team.
    fn buchholz(&self, team: TeamSeed) -> i8 {
        const ONE: Simd<u16, 16> = Simd::splat(1);

        let mask = {
            let shifted = self.opponents[team as usize].splat() >> Self::SEED_LANES;
            (shifted & ONE).cast::<i8>().neg()
        };

        (Simd::from_array(self.diffs) & mask).reduce_sum()
    }

    /// Return a vec of team indices sorted by mid-stage seed calculation.
    ///
    /// 1. Current win-loss record
    /// 2. Buchholz difficulty score (sum of win-loss record for each opponent faced)
    /// 3. Initial seeding
    ///
    /// [Rules and Regs - Mid-stage Seed Calculation](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#Mid-Stage-Seed-Calculation)
    pub(super) fn seed_teams(&self) -> ArrayVec<TeamSeed, 16> {
        // Bitpack seeding information into a 16 bit unsigned integer:
        // [15] [14 13 12 11 10] [9 8 7 6 5] [4 3 2 1 0]
        //  --   --------------   ---------   ----------
        //   |          |             |            |
        // Spare bit    |    Buchholz difficulty   |
        //          Win-loss                 Initial seed
        const FIFTEEN: Simd<i8, 16> = Simd::splat(15);
        let buchholz_array = std::array::from_fn(|i| self.buchholz(i as TeamSeed));
        let buchholz = (FIFTEEN - Simd::from_array(buchholz_array)).cast::<u16>();
        let diffs = (FIFTEEN - Simd::from_array(self.diffs)).cast::<u16>();
        let packed = (diffs << 10 | buchholz << 5 | Self::SEED_LANES).to_array();

        // Select only teams that remain in the tournament.
        let mut seeding = ArrayVec::<_, 16>::new();

        for seed in self.remaining.iter() {
            seeding.push(packed[seed as usize] as TeamSeed);
        }

        seeding.sort_unstable();

        // Strip back down to just the seed.
        for seed in &mut seeding {
            *seed &= 0x1F;
        }

        seeding
    }

    /// Simulate independent match.
    fn simulate_match(&mut self, rng: &mut RngType, seed_a: TeamSeed, seed_b: TeamSeed) {
        let r = rng.random();
        let a = seed_a as usize;
        let b = seed_b as usize;

        // BO3 if match is for advancement/elimination.
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

        // Advance/eliminate teams after BO3.
        if is_bo3 {
            if self.wins[a] == 3 || self.losses[a] == 3 {
                self.remaining.remove(&seed_a);
            }

            if self.wins[b] == 3 || self.losses[b] == 3 {
                self.remaining.remove(&seed_b);
            }
        }
    }

    /// Simulate tournament round.
    #[inline(always)]
    fn simulate_round(&mut self, rng: &mut RngType) {
        for (a, b) in MatchupGenerator::new(&*self) {
            self.simulate_match(rng, a, b);
        }

        self.rounds_complete += 1;
    }

    /// Simulate entire tournament.
    pub fn simulate_tournament(&mut self, rng: &mut RngType) {
        while self.rounds_complete < 5 {
            self.simulate_round(rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate::{Simulation, make_deterministic_rng};

    /// Regression test, will break if the seeding algorithm changes.
    #[test]
    fn regression_test() {
        let sim = Simulation::dummy(1);
        let mut ss = SwissSystem::new(sim.ratings, sim.sigma);
        ss.simulate_tournament(&mut make_deterministic_rng());

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
