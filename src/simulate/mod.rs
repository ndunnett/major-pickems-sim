use std::{
    fmt::Debug,
    iter::Zip,
    ops::{Add, Neg, Range},
    path::PathBuf,
    simd::StdFloat,
    time::{Duration, Instant},
};

use arrayvec::ArrayVec;
use itertools::Itertools;
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::simd::prelude::*;

use crate::data::{TeamSeed, parse_toml};

mod team_set;
use team_set::TeamSet;

type RngType = rand_chacha::ChaCha8Rng;

fn make_rng() -> RngType {
    rand_chacha::ChaCha8Rng::from_rng(&mut rand::rng())
}

fn make_deterministic_rng() -> RngType {
    rand_chacha::ChaCha8Rng::seed_from_u64(7355608)
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

#[derive(Debug)]
enum Matchups {
    Range(Zip<Range<TeamSeed>, Range<TeamSeed>>),
    Vec {
        matchups: ArrayVec<(TeamSeed, TeamSeed), 8>,
        index: usize,
    },
    Iterative {
        teams: ArrayVec<TeamSeed, 16>,
        matchups: ArrayVec<(TeamSeed, TeamSeed), 8>,
        team_index: usize,
        matchup_index: usize,
    },
}

#[derive(Debug)]
struct MatchupGenerator {
    matchups: Matchups,
    opponents: [TeamSet; 16],
    diffs: [i8; 16],
}

impl MatchupGenerator {
    /// Pre-determined matchup priority for a group size of 4.
    const MATCHUP_PRIORITY_4: [[(usize, usize); 2]; 3] = [
        [(0, 3), (1, 2)], // first priority
        [(0, 2), (1, 3)],
        [(0, 1), (2, 3)],
    ];

    /// Pre-determined matchup priority for a group size of 6.
    ///
    /// 0 -> lowest seeded team in the group, 5 -> highest seeded team in the group
    ///
    /// [Rules and Regs - Swiss Bracket](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket)
    const MATCHUP_PRIORITY_6: [[(usize, usize); 3]; 15] = [
        [(0, 5), (1, 4), (2, 3)], // first priority
        [(0, 5), (1, 3), (2, 4)],
        [(0, 4), (1, 5), (2, 3)],
        [(0, 4), (1, 3), (2, 5)],
        [(0, 3), (1, 5), (2, 4)],
        [(0, 3), (1, 4), (2, 5)],
        [(0, 5), (1, 2), (3, 4)],
        [(0, 4), (1, 2), (3, 5)],
        [(0, 2), (1, 5), (3, 4)],
        [(0, 2), (1, 4), (3, 5)],
        [(0, 3), (1, 2), (4, 5)],
        [(0, 2), (1, 3), (4, 5)],
        [(0, 1), (2, 5), (3, 4)],
        [(0, 1), (2, 4), (3, 5)],
        [(0, 1), (2, 3), (4, 5)], // last priority
    ];

    /// Pre-determined matchup priority for a group size of 8.
    ///
    /// Determined by matching highest seed teams first with lowest seed teams.
    /// No need to explore every permutation, only the first 3 options for each team.
    const MATCHUP_PRIORITY_8: [[(usize, usize); 4]; 7] = [
        [(0, 7), (1, 6), (2, 5), (3, 4)], // first priority
        [(0, 6), (1, 7), (2, 5), (3, 4)],
        [(0, 5), (1, 7), (2, 6), (3, 4)],
        [(0, 7), (1, 5), (2, 6), (3, 4)],
        [(0, 7), (1, 4), (2, 6), (3, 5)],
        [(0, 7), (1, 6), (2, 4), (3, 5)],
        [(0, 7), (1, 6), (2, 3), (4, 5)], // last priority
    ];

    /// Pre-determined matchups for second round.
    /// Highest vs. lowest mid-stage seed for each group, groups being 0-7 and 8-15
    const SECOND_ROUND_MATCHUPS: [(usize, usize); 8] = [
        (0, 7),
        (1, 6),
        (2, 5),
        (3, 4),
        (8, 15),
        (9, 14),
        (10, 13),
        (11, 12),
    ];

    pub fn new(ss: &SwissSystem) -> Self {
        Self {
            matchups: match ss.rounds_complete {
                // First round is matched up differently (1-9, 2-10, 3-11 etc.)
                0 => Matchups::Range((0..8).zip(8..16)),
                // Second round is trivial to match
                1 => {
                    let teams = ss.seed_teams();
                    let mut matchups = ArrayVec::new();

                    for (ia, ib) in Self::SECOND_ROUND_MATCHUPS {
                        matchups.push((teams[ia], teams[ib]));
                    }

                    Matchups::Vec { matchups, index: 0 }
                }
                _ => Matchups::Iterative {
                    teams: ss.seed_teams(),
                    matchups: ArrayVec::new(),
                    team_index: 0,
                    matchup_index: 0,
                },
            },
            opponents: ss.opponents,
            diffs: ss.diffs,
        }
    }

    /// Apply a matchup priority lookup table to a group and return an iterator of matchups.
    fn apply_priority<const N: usize, const M: usize>(
        opponents: &[TeamSet],
        priority: [[(usize, usize); M]; N],
        group: &[TeamSeed],
    ) -> ArrayVec<(TeamSeed, TeamSeed), 8> {
        'outer: for indices in priority {
            for (ia, ib) in indices {
                if opponents[group[ia] as usize].contains(&group[ib]) {
                    continue 'outer;
                }
            }

            let mut matchups = ArrayVec::new();

            for (ia, ib) in indices {
                matchups.push((group[ia], group[ib]));
            }

            return matchups;
        }

        unreachable!("matchups without rematch not possible")
    }
}

impl Iterator for MatchupGenerator {
    type Item = (TeamSeed, TeamSeed);

    /// Group team indices by record and arrange matchups, highest seed vs lowest seed.
    ///
    /// Rearrange to avoid rematches:
    ///   - in rounds 2 and 3 (group sizes of 4 or 8), the highest seeded team faces the lowest seeded team that doesn't result in a rematch
    ///   - in rounds 4 and 5 (group sizes of 6), follow pre-determined highest priority matchup that doesn't result in a rematch
    ///
    /// [Rules and Regs - Swiss Bracket](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket)
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.matchups {
            Matchups::Range(range) => range.next(),
            Matchups::Vec { matchups, index } => {
                if *index < matchups.len() {
                    let next = matchups[*index];
                    *index += 1;
                    Some(next)
                } else {
                    None
                }
            }
            Matchups::Iterative {
                teams,
                matchups,
                team_index,
                matchup_index,
            } => loop {
                if *matchup_index < matchups.len() {
                    let next = matchups[*matchup_index];
                    *matchup_index += 1;
                    return Some(next);
                } else if *team_index < teams.len() {
                    *matchup_index = 0;

                    // Chunk into groups of win-loss diff.
                    let start = *team_index;
                    let group_diff = self.diffs[teams[start] as usize];
                    *team_index += 1;

                    while *team_index < teams.len()
                        && self.diffs[teams[*team_index] as usize] == group_diff
                    {
                        *team_index += 1;
                    }

                    // Apply matchup priority to group and extend matchups.
                    match *team_index - start {
                        4 => {
                            *matchups = Self::apply_priority(
                                &self.opponents,
                                Self::MATCHUP_PRIORITY_4,
                                &teams[start..*team_index],
                            )
                        }
                        6 => {
                            *matchups = Self::apply_priority(
                                &self.opponents,
                                Self::MATCHUP_PRIORITY_6,
                                &teams[start..*team_index],
                            )
                        }
                        8 => {
                            *matchups = Self::apply_priority(
                                &self.opponents,
                                Self::MATCHUP_PRIORITY_8,
                                &teams[start..*team_index],
                            )
                        }
                        _ => unreachable!("malformed group"),
                    }
                } else {
                    return None;
                }
            },
        }
    }
}

/// Instance of a single swiss system iteration.
#[derive(Debug, Clone, Copy)]
pub struct SwissSystem {
    wins: [u8; 16],
    losses: [u8; 16],
    diffs: [i8; 16],
    opponents: [TeamSet; 16],
    probabilities_bo1: [[f32; 16]; 16],
    probabilities_bo3: [[f32; 16]; 16],
    remaining: TeamSet,
    rounds_complete: u8,
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
    fn seed_teams(&self) -> ArrayVec<TeamSeed, 16> {
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
