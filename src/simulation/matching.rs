use arrayvec::ArrayVec;

use crate::{
    datatypes::{Index, Set},
    simulation::SwissSystem,
};

type PriorityTable = &'static [&'static [(usize, usize)]];

/// Backing state for generated matchups in the current tournament round.
#[derive(Debug, Clone)]
pub struct Matchups {
    pairs: ArrayVec<(Index, Index), 8>,
}

impl Matchups {
    /// Pre-determined matchup priority for a group size of 4.
    const PRIORITY_4: PriorityTable = &[
        &[(0, 3), (1, 2)], // first priority
        &[(0, 2), (1, 3)],
        &[(0, 1), (2, 3)],
    ];

    /// Pre-determined matchup priority for a group size of 6.
    ///
    /// 0 -> lowest seeded team in the group, 5 -> highest seeded team in the group
    ///
    /// [Rules and Regs - Swiss Bracket](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket)
    const PRIORITY_6: PriorityTable = &[
        &[(0, 5), (1, 4), (2, 3)], // first priority
        &[(0, 5), (1, 3), (2, 4)],
        &[(0, 4), (1, 5), (2, 3)],
        &[(0, 4), (1, 3), (2, 5)],
        &[(0, 3), (1, 5), (2, 4)],
        &[(0, 3), (1, 4), (2, 5)],
        &[(0, 5), (1, 2), (3, 4)],
        &[(0, 4), (1, 2), (3, 5)],
        &[(0, 2), (1, 5), (3, 4)],
        &[(0, 2), (1, 4), (3, 5)],
        &[(0, 3), (1, 2), (4, 5)],
        &[(0, 2), (1, 3), (4, 5)],
        &[(0, 1), (2, 5), (3, 4)],
        &[(0, 1), (2, 4), (3, 5)],
        &[(0, 1), (2, 3), (4, 5)], // last priority
    ];

    /// Pre-determined matchup priority for a group size of 8.
    ///
    /// Determined by matching highest seed teams first with lowest seed teams.
    /// No need to explore every permutation, only the first 3 options for each team.
    const PRIORITY_8: PriorityTable = &[
        &[(0, 7), (1, 6), (2, 5), (3, 4)], // first priority
        &[(0, 6), (1, 7), (2, 5), (3, 4)],
        &[(0, 5), (1, 7), (2, 6), (3, 4)],
        &[(0, 7), (1, 5), (2, 6), (3, 4)],
        &[(0, 7), (1, 4), (2, 6), (3, 5)],
        &[(0, 7), (1, 6), (2, 4), (3, 5)],
        &[(0, 7), (1, 6), (2, 3), (4, 5)], // last priority
    ];

    /// Fixed first-round pairings by initial seed index.
    const FIRST_ROUND: [(Index, Index); 8] = [
        (Index::new::<0>(), Index::new::<8>()),
        (Index::new::<1>(), Index::new::<9>()),
        (Index::new::<2>(), Index::new::<10>()),
        (Index::new::<3>(), Index::new::<11>()),
        (Index::new::<4>(), Index::new::<12>()),
        (Index::new::<5>(), Index::new::<13>()),
        (Index::new::<6>(), Index::new::<14>()),
        (Index::new::<7>(), Index::new::<15>()),
    ];

    /// Pre-determined matchups for second round.
    ///
    /// Highest vs. lowest mid-stage seed for each group, groups being 0-7 and
    /// 8-15.
    const SECOND_ROUND: [(usize, usize); 8] = [
        (0, 7),
        (1, 6),
        (2, 5),
        (3, 4),
        (8, 15),
        (9, 14),
        (10, 13),
        (11, 12),
    ];

    #[cfg_attr(feature = "pprof", inline(never))]
    pub fn new(ss: &SwissSystem) -> Self {
        let mut matchups = Self {
            pairs: ArrayVec::new(),
        };

        match ss.rounds_complete {
            // First round is matched up differently (initial seeds 1-9, 2-10, 3-11 etc.)
            0 => {
                matchups.pairs.extend(Self::FIRST_ROUND);
            }
            // Second round has two 8-team groups and no possible rematches,
            // so the lookup table can be applied immediately.
            1 => {
                let mut winners = [Index::new::<0>(); 8];
                let mut losers = [Index::new::<0>(); 8];
                let mut winner_count = 0;
                let mut loser_count = 0;

                for index in Index::iter_all() {
                    if ss.wins[index.to_usize()] == 1 {
                        winners[winner_count] = index;
                        winner_count += 1;
                    } else {
                        losers[loser_count] = index;
                        loser_count += 1;
                    }
                }

                for &(ia, ib) in &Self::SECOND_ROUND[..4] {
                    matchups.pairs.push((winners[ia], winners[ib]));
                }

                for &(ia, ib) in &Self::SECOND_ROUND[4..] {
                    matchups.pairs.push((losers[ia - 8], losers[ib - 8]));
                }
            }
            _ => {
                let teams = ss.seed_teams();
                let mut team_index = 0;

                // Chunk the sorted teams into groups with the same win-loss
                // differential. In rounds 3-5 these valid group sizes are 4,
                // 6, or 8.
                while team_index < teams.len() {
                    let start = team_index;
                    let group_diff = ss.diffs[teams[start].to_usize()];
                    team_index += 1;

                    while team_index < teams.len()
                        && ss.diffs[teams[team_index].to_usize()] == group_diff
                    {
                        team_index += 1;
                    }

                    let group = &teams[start..team_index];

                    // Apply the priority table for the group size. Each table
                    // is ordered from most preferred to least preferred pairing.
                    match group.len() {
                        4 => matchups.apply_priority(ss, Self::PRIORITY_4, group),
                        6 => matchups.apply_priority(ss, Self::PRIORITY_6, group),
                        8 => matchups.apply_priority(ss, Self::PRIORITY_8, group),
                        _ => unreachable!("malformed group"),
                    }
                }
            }
        }

        matchups
    }

    /// Apply a matchup priority table to a record group.
    ///
    /// The first priority row with no rematches is pushed into the round buffer.
    #[cfg_attr(feature = "pprof", inline(never))]
    #[cfg_attr(not(feature = "pprof"), inline)]
    fn apply_priority(&mut self, ss: &SwissSystem, priority: PriorityTable, group: &[Index]) {
        let mut opponents = ArrayVec::<Set, 8>::new();

        for &team in group {
            opponents.push(ss.opponents[team.to_usize()]);
        }

        'outer: for &indices in priority {
            for &(ia, ib) in indices {
                if opponents[ia].contains(group[ib]) {
                    continue 'outer;
                }
            }

            for &(ia, ib) in indices {
                self.pairs.push((group[ia], group[ib]));
            }

            return;
        }

        unreachable!("matchups without rematch not possible")
    }
}

impl IntoIterator for Matchups {
    type Item = (Index, Index);
    type IntoIter = <ArrayVec<Self::Item, 8> as IntoIterator>::IntoIter;

    #[cfg_attr(feature = "pprof", inline(never))]
    fn into_iter(self) -> Self::IntoIter {
        self.pairs.into_iter()
    }
}
