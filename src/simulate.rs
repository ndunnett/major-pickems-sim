use std::{
    fmt::Debug,
    ops::{Add, Index, IndexMut},
    path::PathBuf,
    time::{Duration, Instant},
};

use itertools::Itertools;
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use smallvec::SmallVec;

use crate::data::{Team, parse_toml};

type RngType = rand_chacha::ChaCha8Rng;

fn make_rng() -> RngType {
    rand_chacha::ChaCha8Rng::from_rng(&mut rand::rng())
}

fn make_deterministic_rng() -> RngType {
    rand_chacha::ChaCha8Rng::seed_from_u64(7355608)
}

/// Store data indexed per team.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeamIndex<T: Debug + Clone> {
    data: [T; 16],
}

impl<T: Debug + Clone> TeamIndex<T> {
    /// Create a new index with default values created by calling `factory`.
    pub fn new<F: Fn() -> T>(factory: F) -> Self {
        Self {
            data: std::array::from_fn(|_| factory()),
        }
    }

    /// Create a new index from an iterator of values. May panic if there aren't 16 values.
    pub fn from_iter<I: Iterator<Item = T>>(values: I) -> Self {
        let mut iter = values.into_iter();

        Self {
            data: std::array::from_fn(|_| iter.next().unwrap()),
        }
    }

    /// Returns an iterator of each value.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Returns an iterator of (key, value) tuples given an array of teams.
    pub fn items<'a>(&'a self, teams: &'a [Team; 16]) -> impl Iterator<Item = (&'a Team, &'a T)> {
        teams.iter().zip(self.iter())
    }
}

impl<T: Debug + Clone> Index<&Team> for TeamIndex<T> {
    type Output = T;

    fn index(&self, team: &Team) -> &Self::Output {
        &self.data[team.index()]
    }
}

impl<T: Debug + Clone> IndexMut<&Team> for TeamIndex<T> {
    fn index_mut(&mut self, team: &Team) -> &mut Self::Output {
        &mut self.data[team.index()]
    }
}

impl<T: Debug + Clone> Index<usize> for TeamIndex<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Debug + Clone> IndexMut<usize> for TeamIndex<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Debug + Clone> Index<u8> for TeamIndex<T> {
    type Output = T;

    fn index(&self, index: u8) -> &Self::Output {
        &self.data[index as usize]
    }
}

impl<T: Debug + Clone> IndexMut<u8> for TeamIndex<T> {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        &mut self.data[index as usize]
    }
}

/// High performance set, specifically for teams.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TeamSet {
    data: u16,
}

impl TeamSet {
    pub const fn new() -> Self {
        Self { data: 0 }
    }

    pub const fn full() -> Self {
        Self { data: u16::MAX }
    }

    /// Insert index into the set.
    pub const fn insert(&mut self, index: u8) -> bool {
        let n = 1_u16 << index;

        if self.data & n == 0 {
            self.data |= n;
            true
        } else {
            false
        }
    }

    /// Remove index from the set.
    pub const fn remove(&mut self, index: u8) -> bool {
        let n = 1_u16 << index;

        if self.data & n == 0 {
            false
        } else {
            self.data &= u16::MAX ^ n;
            true
        }
    }

    /// Test if the set contains index.
    pub const fn contains(&self, index: u8) -> bool {
        self.data & 1_u16 << index != 0
    }

    /// Returns an iterator of indices contained within the set.
    pub fn iter(&self) -> impl Iterator<Item = u8> {
        (0..16).filter(|&i| self.contains(i))
    }
}

/// Struct to keep record of wins, losses, and opponents faced for a team.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TeamRecord {
    wins: i8,
    losses: i8,
    pub teams_faced: TeamSet,
}

impl TeamRecord {
    pub const fn new() -> Self {
        Self {
            wins: 0,
            losses: 0,
            teams_faced: TeamSet::new(),
        }
    }

    /// Win-loss record.
    pub const fn diff(&self) -> i8 {
        self.wins - self.losses
    }

    pub fn buchholz(&self, records: &TeamIndex<TeamRecord>) -> i8 {
        self.teams_faced
            .iter()
            .map(|opp| records[opp].diff())
            .sum::<i8>()
    }
}

/// Struct to tally up tournament results for a team.
#[derive(Debug, Clone, Copy)]
pub struct TeamResult {
    pub three_zero: u64,
    pub advanced: u64,
    pub zero_three: u64,
}

impl TeamResult {
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

impl Add for TeamIndex<TeamResult> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let values = self.iter().zip(rhs.iter()).map(|(a, b)| TeamResult {
            three_zero: a.three_zero + b.three_zero,
            advanced: a.advanced + b.advanced,
            zero_three: a.zero_three + b.zero_three,
        });

        Self::from_iter(values)
    }
}

/// Instance of a single swiss system iteration.
#[derive(Debug, Clone)]
pub struct SwissSystem {
    teams: [Team; 16],
    records: TeamIndex<TeamRecord>,
    probabilities: TeamIndex<TeamIndex<f64>>,
    remaining: TeamSet,
    rounds_complete: u8,
}

impl SwissSystem {
    pub fn new(teams: [Team; 16], sigma: f64) -> Self {
        let records = TeamIndex::new(TeamRecord::new);

        // Precalculate win probabilities for all possible matchups.
        let probabilities = TeamIndex::from_iter(teams.iter().map(|a| {
            TeamIndex::from_iter(
                teams
                    .iter()
                    .map(|b| 1.0 / (1.0 + 10_f64.powf((b.rating - a.rating) as f64 / sigma))),
            )
        }));

        Self {
            teams,
            records,
            probabilities,
            remaining: TeamSet::full(),
            rounds_complete: 0,
        }
    }

    /// Reset Swiss System state to restart tournament.
    pub fn reset(&mut self) {
        let records = TeamIndex::new(TeamRecord::new);
        let remaining = TeamSet::full();
        self.records = records;
        self.remaining = remaining;
        self.rounds_complete = 0;
    }

    /// Return iterator of team indices sorted by mid-stage seed calculation.
    ///
    /// 1. Current win-loss record
    /// 2. Buchholz difficulty score (sum of win-loss record for each opponent faced)
    /// 3. Initial seeding
    ///
    /// [Rules and Regs - Mid-stage Seed Calculation](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#Mid-Stage-Seed-Calculation)
    fn seed_teams(&self) -> impl Iterator<Item = u8> {
        self.remaining.iter().sorted_by_key(|&i| {
            (
                -self.records[i].diff(),
                -self.records[i].buchholz(&self.records),
            )
        })
    }

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

    /// Apply a matchup priority lookup table to a group and return an iterator of matchups.
    fn apply_priority<const N: usize, const M: usize>(
        &self,
        priority: [[(usize, usize); M]; N],
        group: SmallVec<u8, 8>,
    ) -> impl Iterator<Item = (u8, u8)> {
        for indices in priority {
            if indices
                .iter()
                .all(|(ia, ib)| !self.records[group[*ia]].teams_faced.contains(group[*ib]))
            {
                return indices
                    .into_iter()
                    .map(move |(ia, ib)| (group[ia], group[ib]));
            }
        }

        unreachable!("matchups without rematch not possible")
    }

    /// Group team indices by record and arrange matchups, highest seed vs lowest seed.
    ///
    /// Rearrange to avoid rematches:
    ///   - in rounds 2 and 3 (group sizes of 4 or 8), the highest seeded team faces the lowest seeded team that doesn't result in a rematch
    ///   - in rounds 4 and 5 (group sizes of 6), follow pre-determined highest priority matchup that doesn't result in a rematch
    ///
    /// [Rules and Regs - Swiss Bracket](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket)
    fn generate_matchups(&self) -> SmallVec<(u8, u8), 8> {
        self.seed_teams()
            .chunk_by(|&index| self.records[index].diff())
            .into_iter()
            .fold(SmallVec::new(), |mut acc, (_, group_iter)| {
                let group = group_iter.collect::<SmallVec<_, 8>>();

                match group.len() {
                    4 => acc.extend(self.apply_priority(Self::MATCHUP_PRIORITY_4, group)),
                    6 => acc.extend(self.apply_priority(Self::MATCHUP_PRIORITY_6, group)),
                    8 => acc.extend(self.apply_priority(Self::MATCHUP_PRIORITY_8, group)),
                    _ => unreachable!("malformed group"),
                }

                acc
            })
    }

    /// Simulate independent match.
    fn simulate_match(&mut self, rng: &mut RngType, index_a: u8, index_b: u8) {
        // BO3 if match is for advancement/elimination.
        let is_bo3 = self.records[index_a].wins == 2 || self.records[index_a].losses == 2;

        // Simulate match outcome.
        let p = self.probabilities[index_a][index_b];

        let team_a_win = if is_bo3 {
            let first_map = p > rng.random();
            let second_map = p > rng.random();

            if first_map != second_map {
                p > rng.random()
            } else {
                first_map
            }
        } else {
            p > rng.random()
        };

        // Update team records.
        if team_a_win {
            self.records[index_a].wins += 1;
            self.records[index_b].losses += 1;
        } else {
            self.records[index_a].losses += 1;
            self.records[index_b].wins += 1;
        }

        self.records[index_a].teams_faced.insert(index_b);
        self.records[index_b].teams_faced.insert(index_a);

        // Advance/eliminate teams after BO3.
        if is_bo3 {
            for index in [index_a, index_b] {
                if self.records[index].wins == 3 || self.records[index].losses == 3 {
                    self.remaining.remove(index);
                }
            }
        }
    }

    /// Simulate tournament round.
    fn simulate_round(&mut self, rng: &mut RngType) {
        match self.rounds_complete {
            0 => {
                // First round is matched up differently (1-9, 2-10, 3-11 etc.)
                for (index_a, index_b) in (0..8).zip(8..16) {
                    self.simulate_match(rng, index_a, index_b);
                }
            }
            1 => {
                // Second round is trivial to match
                let teams = self.seed_teams().collect::<SmallVec<u8, 16>>();

                for (seed_a, seed_b) in Self::SECOND_ROUND_MATCHUPS {
                    self.simulate_match(rng, teams[seed_a], teams[seed_b]);
                }
            }
            _ => {
                // Remaining rounds can be matched with lookup tables
                for (index_a, index_b) in self.generate_matchups() {
                    self.simulate_match(rng, index_a, index_b);
                }
            }
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
    teams: [Team; 16],
}

impl Simulation {
    pub fn try_from_file(filepath: PathBuf) -> anyhow::Result<Self> {
        Ok(Self {
            teams: parse_toml(filepath)?,
        })
    }

    /// Produce simulation with dummy data for testing purposes.
    pub fn dummy() -> Self {
        Self {
            teams: (0..16)
                .map(|i| Team {
                    name: format!("Team {}", i + 1),
                    seed: i as u8 + 1,
                    rating: 2000 - 50 * i,
                })
                .collect_array()
                .unwrap(),
        }
    }

    /// Run single-threaded bench test for profiling/benchmarking purposes.
    pub fn bench_test(iterations: usize) -> TeamIndex<TeamResult> {
        let sim = Self::dummy();
        let mut results = TeamIndex::new(TeamResult::new);
        let fresh_ss = SwissSystem::new(sim.teams.clone(), 800.0);
        let mut rng = make_deterministic_rng();

        for _ in 0..iterations {
            let mut ss = fresh_ss.clone();
            ss.simulate_tournament(&mut rng);

            for (team, record) in ss.records.items(&ss.teams) {
                match (record.wins, record.losses) {
                    (3, 0) => results[team].three_zero += 1,
                    (3, 1 | 2) => results[team].advanced += 1,
                    (0, 3) => results[team].zero_three += 1,
                    _ => {}
                }
            }
        }

        results
    }

    /// Run 'n' iterations of tournament simulation.
    pub fn run(&self, n: u64, sigma: f64) -> TeamIndex<TeamResult> {
        let fresh_ss = SwissSystem::new(self.teams.clone(), sigma);
        let fresh_results = TeamIndex::new(TeamResult::new);

        (0..n)
            .into_par_iter()
            .map_init(
                || (fresh_ss.clone(), make_rng()),
                |(ss, rng), _| {
                    let mut results = fresh_results;
                    ss.reset();
                    ss.simulate_tournament(rng);

                    for (team, record) in ss.records.items(&ss.teams) {
                        match (record.wins, record.losses) {
                            (3, 0) => results[team].three_zero += 1,
                            (3, 1 | 2) => results[team].advanced += 1,
                            (0, 3) => results[team].zero_three += 1,
                            _ => {}
                        }
                    }

                    results
                },
            )
            .reduce(|| fresh_results, |acc, result| acc + result)
    }

    /// Format results from a simulation into a readable/printable string.
    pub fn format_results(
        &self,
        results: TeamIndex<TeamResult>,
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
            let sorted_results = results
                .items(&self.teams)
                .sorted_by(|(_, a), (_, b)| func(b).cmp(&func(a)))
                .enumerate();

            // Format each result into a string.
            for (i, (team, result)) in sorted_results {
                out.push(format!(
                    "{num:<4}{name:<20}{percent:>6.1}%",
                    num = format!("{}.", i + 1),
                    name = team.name,
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
pub fn simulate(file: PathBuf, iterations: u64, sigma: f64) -> anyhow::Result<()> {
    let now = Instant::now();
    let sim = Simulation::try_from_file(file)?;
    let results = sim.run(iterations, sigma);
    println!("{}", sim.format_results(results, iterations, now.elapsed()));
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
            (0..16_u8)
                .map(|index| results[index].three_zero as usize)
                .sum::<usize>(),
            iterations * 2
        );

        // Total 3-1/3-2 stats should sum to 6 per iteration
        assert_eq!(
            (0..16_u8)
                .map(|index| results[index].advanced as usize)
                .sum::<usize>(),
            iterations * 6
        );

        // Total 0-3 stats should sum to 2 per iteration
        assert_eq!(
            (0..16_u8)
                .map(|index| results[index].zero_three as usize)
                .sum::<usize>(),
            iterations * 2
        );

        // Best team should always have more 3-0 stats than the worst team
        assert!(results.data[0].three_zero > results.data[15].three_zero);

        // Best team should always have less 0-3 stats than the worst team
        assert!(results.data[0].zero_three < results.data[15].zero_three);
    }

    /// Regression test, will break if the seeding algorithm changes.
    #[test]
    fn regression_test() {
        let mut rng = make_deterministic_rng();
        let sim = Simulation::dummy();
        let mut ss = SwissSystem::new(sim.teams.clone(), 800.0);
        ss.simulate_tournament(&mut rng);

        let expected_records = TeamIndex {
            data: [
                TeamRecord {
                    wins: 3,
                    losses: 1,
                    teams_faced: TeamSet { data: 21248 },
                },
                TeamRecord {
                    wins: 3,
                    losses: 0,
                    teams_faced: TeamSet { data: 648 },
                },
                TeamRecord {
                    wins: 3,
                    losses: 1,
                    teams_faced: TeamSet { data: 5312 },
                },
                TeamRecord {
                    wins: 3,
                    losses: 1,
                    teams_faced: TeamSet { data: 2146 },
                },
                TeamRecord {
                    wins: 2,
                    losses: 3,
                    teams_faced: TeamSet { data: 45632 },
                },
                TeamRecord {
                    wins: 1,
                    losses: 3,
                    teams_faced: TeamSet { data: 10376 },
                },
                TeamRecord {
                    wins: 3,
                    losses: 2,
                    teams_faced: TeamSet { data: 17436 },
                },
                TeamRecord {
                    wins: 3,
                    losses: 2,
                    teams_faced: TeamSet { data: 33830 },
                },
                TeamRecord {
                    wins: 0,
                    losses: 3,
                    teams_faced: TeamSet { data: 24577 },
                },
                TeamRecord {
                    wins: 2,
                    losses: 3,
                    teams_faced: TeamSet { data: 10259 },
                },
                TeamRecord {
                    wins: 2,
                    losses: 3,
                    teams_faced: TeamSet { data: 18628 },
                },
                TeamRecord {
                    wins: 3,
                    losses: 2,
                    teams_faced: TeamSet { data: 34344 },
                },
                TeamRecord {
                    wins: 3,
                    losses: 0,
                    teams_faced: TeamSet { data: 21 },
                },
                TeamRecord {
                    wins: 1,
                    losses: 3,
                    teams_faced: TeamSet { data: 816 },
                },
                TeamRecord {
                    wins: 1,
                    losses: 3,
                    teams_faced: TeamSet { data: 1345 },
                },
                TeamRecord {
                    wins: 0,
                    losses: 3,
                    teams_faced: TeamSet { data: 2192 },
                },
            ],
        };

        assert_eq!(ss.records, expected_records);
    }
}
