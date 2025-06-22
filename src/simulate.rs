use std::{
    collections::HashSet,
    fmt::Debug,
    ops::{Add, Index, IndexMut},
    path::PathBuf,
    time::Duration,
};

use itertools::Itertools;
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::data::{Team, parse_toml};

/// Pre-determined matchup priority for a group size of 6.
///
/// 0 -> lowest seeded team in the group, 5 -> highest seeded team in the group
///
/// [Rules and Regs - Swiss Bracket](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket)
const MATCHUP_PRIORITY: [[(usize, usize); 3]; 15] = [
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

/// Calculate the probability of team 'a' beating team 'b' for given sigma value.
fn win_probability(a: &Team, b: &Team, sigma: f64) -> f64 {
    1.0 / (1.0 + 10_f64.powf((b.rating - a.rating) as f64 / sigma))
}

/// Store data indexed per team.
#[derive(Debug, Clone)]
pub struct TeamIndex<T: Debug + Clone> {
    teams: Vec<Team>,
    data: Vec<T>,
}

impl<T: Debug + Clone> TeamIndex<T> {
    /// Create a new index from an iterator of teams, with default values created by calling `factory`.
    pub fn new<I: IntoIterator<Item = Team>, F: FnOnce() -> T>(teams: I, factory: F) -> Self {
        let teams = Vec::from_iter(teams);
        let data = vec![factory(); teams.len()];
        Self { teams, data }
    }

    /// Returns an iterator of each key (team).
    pub fn keys(&self) -> impl Iterator<Item = &Team> {
        self.teams.iter()
    }

    /// Returns an iterator of each value.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Returns an iterator of (key, value) tuples.
    pub fn items(&self) -> impl Iterator<Item = (&Team, &T)> + '_ {
        self.keys().zip(self.values())
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

/// Struct to keep record of wins, losses, and opponents faced for a team.
#[derive(Debug, Clone)]
struct TeamRecord {
    wins: i8,
    losses: i8,
    pub teams_faced: HashSet<Team>,
}

impl TeamRecord {
    pub fn new() -> Self {
        Self {
            wins: 0,
            losses: 0,
            teams_faced: HashSet::new(),
        }
    }

    /// Win-loss record.
    pub fn diff(&self) -> i8 {
        self.wins - self.losses
    }
}

/// Struct to tally up tournament results for a team.
#[derive(Debug, Clone)]
pub struct TeamResult {
    pub three_zero: u64,
    pub advanced: u64,
    pub zero_three: u64,
}

impl TeamResult {
    pub fn new() -> Self {
        Self {
            three_zero: 0,
            advanced: 0,
            zero_three: 0,
        }
    }
}

impl Add for TeamResult {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            three_zero: self.three_zero + rhs.three_zero,
            advanced: self.advanced + rhs.advanced,
            zero_three: self.zero_three + rhs.zero_three,
        }
    }
}

impl Add for TeamIndex<TeamResult> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut new = self.clone();

        for (key, value) in self.items() {
            new[key] = value.clone() + rhs[key].clone();
        }

        new
    }
}

/// Instance of a single swiss system iteration.
#[derive(Debug, Clone)]
pub struct SwissSystem {
    sigma: f64,
    records: TeamIndex<TeamRecord>,
    remaining: HashSet<Team>,
    first_round: bool,
}

impl SwissSystem {
    pub fn new<I: IntoIterator<Item = Team>>(teams: I, sigma: f64) -> Self {
        let records = TeamIndex::new(teams, TeamRecord::new);
        let remaining = HashSet::from_iter(records.keys().cloned());
        let first_round = true;

        Self {
            sigma,
            records,
            remaining,
            first_round,
        }
    }

    fn simulate_match(&mut self, team_a: &Team, team_b: &Team) {
        // BO3 if match is for advancement/elimination
        let is_bo3 = self.records[team_a].wins == 2 || self.records[team_a].losses == 2;

        // Calculate single map win probability
        let p = win_probability(team_a, team_b, self.sigma);

        // Simulate match outcome
        let mut rng = rand::rng();
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

        // Update team records
        if team_a_win {
            self.records[team_a].wins += 1;
            self.records[team_b].losses += 1;
        } else {
            self.records[team_a].losses += 1;
            self.records[team_b].wins += 1;
        }

        // Add to faced teams
        self.records[team_a].teams_faced.insert(team_b.clone());
        self.records[team_b].teams_faced.insert(team_a.clone());

        // Advance/eliminate teams after BO3
        if is_bo3 {
            for team in &[team_a, team_b] {
                if self.records[team].wins == 3 || self.records[team].losses == 3 {
                    self.remaining.remove(team);
                }
            }
        }
    }

    fn simulate_round(&mut self) {
        let mut teams = Vec::from_iter(self.remaining.iter().cloned());

        // Sort teams by mid-stage seed calculation:
        //   1. Current win-loss record
        //   2. Buchholz difficulty score (sum of win-loss record for each opponent faced)
        //   3. Initial seeding
        // https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#Mid-Stage-Seed-Calculation

        teams.sort_by_key(|team| {
            (
                -self.records[team].diff(),
                -self.records[team]
                    .teams_faced
                    .iter()
                    .map(|opp| self.records[opp].diff())
                    .sum::<i8>(),
                team.seed,
            )
        });

        let matchups = if self.first_round {
            // First round is seeded differently (1-9, 2-10, 3-11 etc.)
            self.first_round = false;
            teams
                .iter()
                .zip(teams.iter().skip(teams.len() / 2))
                .map(|(a, b)| (a.clone(), b.clone()))
                .collect()
        } else {
            teams
                .into_iter()
                .chunk_by(|team| self.records[team].diff())
                .into_iter()
                .fold(Vec::new(), |mut acc, (_, group_iter)| {
                    let group = group_iter.collect::<Vec<_>>();

                    // Group teams by record and run matches for each group, highest seed vs lowest seed.
                    // Rearrange to avoid rematches:
                    //   - in rounds 2 and 3 (group sizes of 4 or 8), the highest seeded team faces the lowest seeded team that doesn't result in a rematch
                    //   - in rounds 4 and 5 (group sizes of 6), follow pre-determined highest priority matchup that doesn't result in a rematch

                    if group.len() == 6 {
                        for indices in MATCHUP_PRIORITY {
                            if indices.iter().all(|(ia, ib)| {
                                !self.records[&group[*ia]].teams_faced.contains(&group[*ib])
                            }) {
                                acc.extend(
                                    indices
                                        .iter()
                                        .map(|(ia, ib)| (group[*ia].clone(), group[*ib].clone())),
                                );
                                break;
                            }
                        }
                    } else {
                        'outer: for arrangement in 0..usize::MAX {
                            let mut group = group.clone();

                            // After the first attempt, start rearranging seeding to ensure no rematches.
                            // Start by swapping highest and 2nd highest seeded teams and progress through
                            // on each loop, eventually trying every possible permutation.
                            if arrangement > 0 {
                                let mid_point = group.len() / 2;
                                let multiples = arrangement / mid_point;
                                let remainder = arrangement % mid_point;

                                if multiples * 2 <= mid_point {
                                    for _ in 0..multiples {
                                        group.swap(multiples * 2, multiples * 2 + 1);
                                    }

                                    if remainder > 0 {
                                        group.swap(
                                            multiples * 2 + remainder,
                                            multiples * 2 + remainder + 1,
                                        );
                                    }
                                } else {
                                    // All swaps are exhausted, this should never happen.
                                    panic!("impossible to avoid rematch")
                                }
                            }

                            let half = group.len() / 2;
                            let mut bottom_teams =
                                group.iter().skip(half).cloned().collect::<Vec<_>>();
                            let mut matchups = vec![];

                            // Attempt to assign matchups, continue to the next iteration of the outer loop if it fails.
                            'inner: for team_a in group.iter().take(half) {
                                for i in (0..bottom_teams.len()).rev() {
                                    if !self.records[team_a].teams_faced.contains(&bottom_teams[i])
                                    {
                                        matchups.push((team_a.clone(), bottom_teams.remove(i)));
                                        continue 'inner;
                                    }
                                }

                                continue 'outer;
                            }

                            acc.extend(matchups);
                            break;
                        }
                    }

                    acc
                })
        };

        for (team_a, team_b) in matchups.iter() {
            self.simulate_match(team_a, team_b);
        }
    }

    pub fn simulate_tournament(&mut self) {
        while !self.remaining.is_empty() {
            self.simulate_round();
        }
    }
}

type TeamResultField = fn(&TeamResult) -> u64;

/// Instance of a simulation, to parse team data and run tournament iterations.
#[derive(Debug, Clone)]
pub struct Simulation {
    sigma: f64,
    teams: [Team; 16],
}

impl Simulation {
    pub fn try_from_file(filepath: PathBuf, sigma: f64) -> anyhow::Result<Self> {
        Ok(Self {
            sigma,
            teams: parse_toml(filepath)?,
        })
    }

    /// Run 'n' iterations of tournament simulation.
    pub fn run(&self, n: u64) -> TeamIndex<TeamResult> {
        let fresh_results = TeamIndex::new(self.teams.iter().cloned(), TeamResult::new);

        (0..n)
            .into_par_iter()
            .map(|_| {
                let mut results = fresh_results.clone();
                let mut ss = SwissSystem::new(self.teams.iter().cloned(), self.sigma);
                ss.simulate_tournament();

                for (team, record) in ss.records.items() {
                    match (record.wins, record.losses) {
                        (3, 0) => results[team].three_zero += 1,
                        (3, 1 | 2) => results[team].advanced += 1,
                        (0, 3) => results[team].zero_three += 1,
                        _ => {}
                    }
                }

                results
            })
            .reduce(|| fresh_results.clone(), |acc, result| acc + result)
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
                .items()
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
    let now = std::time::Instant::now();
    let sim = Simulation::try_from_file(file, sigma)?;
    let results = sim.run(iterations);
    println!("{}", sim.format_results(results, iterations, now.elapsed()));
    Ok(())
}
