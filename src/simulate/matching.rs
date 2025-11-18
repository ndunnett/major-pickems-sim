use std::{iter::Zip, ops::Range};

use arrayvec::ArrayVec;

use crate::{
    data::TeamSeed,
    simulate::{SwissSystem, TeamSet},
};

/// Different representations of seeded matchups.
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

/// Struct to iterate seeded matchups.
#[derive(Debug)]
pub struct MatchupGenerator {
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
