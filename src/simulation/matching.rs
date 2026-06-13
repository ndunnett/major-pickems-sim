use crate::{
    datatypes::Index,
    simulation::{SwissSystem, seed_teams},
};

type PriorityTable = &'static [&'static [(usize, usize)]];

/// Backing state for generated matchups in the current tournament round.
#[derive(Debug, Clone)]
pub struct Matchups {
    pairs: [(Index, Index); 8],
    len: usize,
    index: usize,
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

    const PRIORITY_TABLES: &[PriorityTable; 9] = &[
        &[],
        &[],
        &[],
        &[],
        Self::PRIORITY_4,
        &[],
        Self::PRIORITY_6,
        &[],
        Self::PRIORITY_8,
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
    #[must_use]
    pub fn new(ss: &SwissSystem) -> Self {
        let mut matchups = Self {
            pairs: [(Index::new::<0>(), Index::new::<0>()); 8],
            len: 0,
            index: 0,
        };

        match ss.rounds_complete() {
            // First round is matched up differently (initial seeds 1-9, 2-10, 3-11 etc.)
            0 => {
                matchups.pairs = Self::FIRST_ROUND;
                matchups.len = 8;
            }
            // Second round has two 8-team groups and no possible rematches,
            // so the lookup table can be applied immediately.
            1 => {
                let mut winners = [Index::new::<0>(); 8];
                let mut losers = [Index::new::<0>(); 8];
                let mut winner_count = 0;
                let mut loser_count = 0;

                for index in Index::iter_all() {
                    if ss.wins(index) == 1 {
                        winners[winner_count] = index;
                        winner_count += 1;
                    } else {
                        losers[loser_count] = index;
                        loser_count += 1;
                    }
                }

                for &(ia, ib) in &Self::SECOND_ROUND[..4] {
                    // SAFETY: second round always produces eight total matchups.
                    unsafe {
                        matchups.push_unchecked((winners[ia], winners[ib]));
                    }
                }

                for &(ia, ib) in &Self::SECOND_ROUND[4..] {
                    // SAFETY: second round always produces eight total matchups.
                    unsafe {
                        matchups.push_unchecked((losers[ia - 8], losers[ib - 8]));
                    }
                }
            }
            2 => {
                let teams = seed_teams(ss.remaining(), ss.diffs(), ss.all_opponents());
                matchups.apply_priority(ss, &teams[0..4]);
                matchups.apply_priority(ss, &teams[4..12]);
                matchups.apply_priority(ss, &teams[12..16]);
            }
            3 => {
                let teams = seed_teams(ss.remaining(), ss.diffs(), ss.all_opponents());
                matchups.apply_priority(ss, &teams[0..6]);
                matchups.apply_priority(ss, &teams[6..12]);
            }
            4 => {
                let teams = seed_teams(ss.remaining(), ss.diffs(), ss.all_opponents());
                matchups.apply_priority(ss, &teams[0..6]);
            }
            5 => {}
            _ => unreachable!("Swiss simulation has only five rounds"),
        }

        matchups
    }

    /// Push a generated matchup into the fixed round buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no more than 8 elements are pushed.
    #[inline]
    unsafe fn push_unchecked(&mut self, pair: (Index, Index)) {
        debug_assert!(self.len < self.pairs.len());
        unsafe { *self.pairs.get_unchecked_mut(self.len) = pair };
        self.len += 1;
    }

    /// Apply a matchup priority table to a record group.
    ///
    /// The first priority row with no rematches is pushed into the round buffer.
    #[cfg_attr(feature = "pprof", inline(never))]
    #[cfg_attr(not(feature = "pprof"), inline)]
    fn apply_priority(&mut self, ss: &SwissSystem, group: &[Index]) {
        let priority = Self::PRIORITY_TABLES[group.len()];
        let mut opponent_bits = [0; 8];
        let mut team_bits = [0; 8];

        for (i, &team) in group.iter().enumerate() {
            opponent_bits[i] = ss.opponents(team).to_bits();
            team_bits[i] = team.bit_select();
        }

        'outer: for &indices in priority {
            for &(ia, ib) in indices {
                // SAFETY: Priority tables never index past the 8th element.
                if unsafe { *opponent_bits.get_unchecked(ia) & *team_bits.get_unchecked(ib) } != 0 {
                    continue 'outer;
                }
            }

            for &(ia, ib) in indices {
                // SAFETY: Swiss record groups produce at most eight total matchups, and the caller
                // guarantees that the priority table won't index out of the bounds of `group`.
                unsafe {
                    self.push_unchecked((*group.get_unchecked(ia), *group.get_unchecked(ib)));
                }
            }

            return;
        }

        unreachable!("matchups without rematch not possible")
    }
}

impl ExactSizeIterator for Matchups {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl Iterator for Matchups {
    type Item = (Index, Index);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let ret = Some(unsafe { *self.pairs.get_unchecked(self.index) });
            self.index += 1;
            ret
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.len - self.index;
        (rem, Some(rem))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensure that priority tables won't index out of bounds for their group size.
    #[test]
    fn priority_table_indices() {
        for (priority, len) in [
            (Matchups::PRIORITY_4, 4),
            (Matchups::PRIORITY_6, 6),
            (Matchups::PRIORITY_8, 8),
        ] {
            for &row in priority {
                for &(a, b) in row {
                    assert!(a < len);
                    assert!(b < len);
                }
            }
        }
    }
}
