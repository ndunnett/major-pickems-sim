use std::ops::Range;

use crate::{
    datatypes::Index,
    simulation::{SwissSystem, seed_teams},
};

type Pair = (Index, Index);
type Pairs = [Pair; 8];

const EMPTY_PAIRS: Pairs = [(Index::new::<0>(), Index::new::<0>()); 8];

/// Fixed first-round pairings by initial seed index.
const FIRST_ROUND: Pairs = [
    (Index::new::<0>(), Index::new::<8>()),
    (Index::new::<1>(), Index::new::<9>()),
    (Index::new::<2>(), Index::new::<10>()),
    (Index::new::<3>(), Index::new::<11>()),
    (Index::new::<4>(), Index::new::<12>()),
    (Index::new::<5>(), Index::new::<13>()),
    (Index::new::<6>(), Index::new::<14>()),
    (Index::new::<7>(), Index::new::<15>()),
];

/// Look up table for second round matchups based on all possible first round outcomes.
/// Bit `i` of the key is true if the first team in `FIRST_ROUND[i]` (the one with
/// the lower seed number) won, or false if the second team won.
///
/// Each entry lists the winners' four matches first, then the losers' four.
/// Within each group the highest seed plays the lowest, and so on.
///
/// # Examples
///
/// - `SECOND_ROUND_LUT[0b0000_0001]`: seed 1 won, seeds 2-8 lost
/// - `SECOND_ROUND_LUT[0b0001_0000]`: seed 5 won, seeds 1-4 and 6-8 lost
const SECOND_ROUND_LUT: [Pairs; 256] = {
    let mut table = [EMPTY_PAIRS; 256];
    let mut key = 0;

    while key < 256 {
        let mut seeded = [Index::new::<0>(); 16];
        let mut winner = 0;
        let mut loser = 8;
        let mut seed = 0;

        while seed < 16 {
            let i = seed % 8;
            let high = seed < 8;
            let (a, b) = FIRST_ROUND[i];
            let team = if high { a } else { b };
            let high_wins = key & (1 << i) != 0;

            if high == high_wins {
                seeded[winner] = team;
                winner += 1;
            } else {
                seeded[loser] = team;
                loser += 1;
            }

            seed += 1;
        }

        let mut i = 0;

        while i < 4 {
            table[key][i] = (seeded[i], seeded[7 - i]);
            table[key][4 + i] = (seeded[8 + i], seeded[15 - i]);
            i += 1;
        }

        key += 1;
    }

    table
};

/// Ordered pairing choices for `TEAMS` Swiss teams with the same win-loss record.
///
/// Each row pairs all teams in `MATCHES` matches using their positions in the seeded
/// group. For example, `(0, 5)` pairs the group's first seed with its sixth.
/// The first row that contains no rematches determines the draw.
struct PriorityTable<const TEAMS: usize, const MATCHES: usize> {
    rows: &'static [[(usize, usize); MATCHES]],
    /// Bit `i` in `rows_with[a][b]` is set when `rows[i]` matches `a` with `b`.
    rows_with: [[u16; TEAMS]; TEAMS],
}

impl<const TEAMS: usize, const MATCHES: usize> PriorityTable<TEAMS, MATCHES> {
    /// Builds the pair-to-row masks for the priority table.
    const fn new(rows: &'static [[(usize, usize); MATCHES]]) -> Self {
        assert!(
            !rows.is_empty() && rows.len() <= 15,
            "PriorityTable can have no more than 15 rows and must not be empty"
        );

        let mut rows_with = [[0; TEAMS]; TEAMS];
        let mut i = 0;

        while i < rows.len() {
            let mut j = 0;

            while j < MATCHES {
                let (a, b) = rows[i][j];
                // A rematch rules out this row regardless of which team is listed first.
                rows_with[a][b] |= 1 << i;
                rows_with[b][a] |= 1 << i;
                j += 1;
            }

            i += 1;
        }

        Self { rows, rows_with }
    }

    /// Returns the first rematch free draw in row order, with group positions
    /// translated to team indices. `group` must contain `TEAMS` teams in seed order.
    #[inline]
    fn match_group(&self, ss: &SwissSystem, group: &[Index]) -> [Pair; MATCHES] {
        // Start with one set bit per row. The lowest set bit is the next row to try.
        let mut eligible = (1_u16 << self.rows.len()) - 1;

        'candidates: while eligible != 0 {
            let row = &self.rows[eligible.trailing_zeros() as usize];

            for &(a, b) in row {
                if ss.opponents(group[a]).contains(group[b]) {
                    // Every row containing this pair would repeat the same match.
                    eligible &= !self.rows_with[a][b];
                    continue 'candidates;
                }
            }

            return row.map(|(a, b)| (group[a], group[b]));
        }

        unreachable!("matchups without rematch not possible")
    }
}

/// Pre-determined matchup priority for a group size of 6 according to
/// [Swiss Bracket] rules.
///
/// [Swiss Bracket]: https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket
const PRIORITY_6: PriorityTable<6, 3> = PriorityTable::new(&[
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
    [(0, 1), (2, 3), (4, 5)],
]);

/// Pre-determined matchup priority for a group size of 8.
///
/// Determined by matching highest seed teams first with lowest seed teams.
const PRIORITY_8: PriorityTable<8, 4> = PriorityTable::new(&[
    [(0, 7), (1, 6), (2, 5), (3, 4)], // first priority
    [(0, 7), (1, 6), (2, 4), (3, 5)],
    [(0, 7), (1, 5), (2, 6), (3, 4)],
    [(0, 7), (1, 5), (2, 4), (3, 6)],
    [(0, 6), (1, 7), (2, 5), (3, 4)],
    [(0, 6), (1, 7), (2, 4), (3, 5)],
]);

/// Backing state for generated matchups in the current tournament round.
#[derive(Debug, Clone)]
pub struct Matchups {
    pairs: Pairs,
    remaining: Range<usize>,
}

impl Matchups {
    #[must_use]
    pub fn new(ss: &SwissSystem) -> Self {
        match ss.rounds_complete() {
            // First round is matched up differently (initial seeds 1-9, 2-10, 3-11 etc.)
            0 => Self {
                pairs: FIRST_ROUND,
                remaining: 0..8,
            },
            // Use the lookup table with the first round outcome to get the second round matchups
            1 => {
                // Byte i is the win count (0 or 1) of the first team in FIRST_ROUND[i]
                let wins = u64::from_le_bytes(ss.first_8_wins());

                // Multiply to gather each byte's low bit into the top byte, shift the top byte
                // down to the low byte such that match i's outcome is in bit i of the key
                let key = (wins.wrapping_mul(0x0102_0408_1020_4080) >> 56) as usize;

                Self {
                    pairs: SECOND_ROUND_LUT[key],
                    remaining: 0..8,
                }
            }
            // Rounds 3-5 are determined by applying priority tables and filtering out rematches
            2 => {
                let teams = seed_teams(ss.remaining(), ss.diffs(), ss.all_opponents());
                let mut pairs = EMPTY_PAIRS;
                pairs[2..6].copy_from_slice(&PRIORITY_8.match_group(ss, &teams[4..12]));

                // Two 2–0 teams cannot have played one another; likewise for two 0–2 teams.
                pairs[0] = (teams[0], teams[3]);
                pairs[1] = (teams[1], teams[2]);
                pairs[6] = (teams[12], teams[15]);
                pairs[7] = (teams[13], teams[14]);

                Self {
                    pairs,
                    remaining: 0..8,
                }
            }
            3 => {
                let teams = seed_teams(ss.remaining(), ss.diffs(), ss.all_opponents());
                let mut pairs = EMPTY_PAIRS;
                pairs[..3].copy_from_slice(&PRIORITY_6.match_group(ss, &teams[..6]));
                pairs[3..6].copy_from_slice(&PRIORITY_6.match_group(ss, &teams[6..12]));

                Self {
                    pairs,
                    remaining: 0..6,
                }
            }
            4 => {
                let teams = seed_teams(ss.remaining(), ss.diffs(), ss.all_opponents());
                let mut pairs = EMPTY_PAIRS;
                pairs[..3].copy_from_slice(&PRIORITY_6.match_group(ss, &teams[..6]));

                Self {
                    pairs,
                    remaining: 0..3,
                }
            }
            _ => unreachable!("Swiss simulation has only five rounds"),
        }
    }
}

impl ExactSizeIterator for Matchups {
    #[inline]
    fn len(&self) -> usize {
        self.remaining.len()
    }
}

impl Iterator for Matchups {
    type Item = Pair;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.remaining
            .next()
            .map(|i| unsafe { *self.pairs.get_unchecked(i) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.remaining.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Reverse;

    use super::*;
    use crate::{
        datatypes::{Rating, Set, Sigma, Teams},
        simulation::rng,
    };

    /// Compare ordered pairings to a linear row scan to verify that [`PriorityTable`] applies
    /// priority correctly according to the given table.
    #[test]
    fn priority_table_follows_order() {
        const FOUR: &[[(usize, usize); 2]] =
            &[[(0, 3), (1, 2)], [(0, 2), (1, 3)], [(0, 1), (2, 3)]];

        fn scan<const N: usize>(
            ss: &SwissSystem,
            group: &[Index],
            rows: &[[(usize, usize); N]],
        ) -> [Pair; N] {
            rows.iter()
                .map(|row| row.map(|(a, b)| (group[a], group[b])))
                .find(|pairs| pairs.iter().all(|&(a, b)| !ss.opponents(a).contains(b)))
                .unwrap()
        }

        let fresh = SwissSystem::new(Teams::dummy().ratings, Sigma::default());
        let mut rng = rng::deterministic();

        for _ in 0..256 {
            let mut ss = fresh;
            ss.simulate_round(&mut rng);
            ss.simulate_round(&mut rng);

            for round in 3..=5 {
                let teams = seed_teams(ss.remaining(), ss.diffs(), ss.all_opponents());
                let mut expected = Vec::new();

                if round == 3 {
                    expected.extend(scan(&ss, &teams[..4], FOUR));
                    expected.extend(scan(&ss, &teams[4..12], PRIORITY_8.rows));
                    expected.extend(scan(&ss, &teams[12..], FOUR));
                } else {
                    for group in teams.as_chunks::<6>().0 {
                        expected.extend(scan(&ss, group, PRIORITY_6.rows));
                    }
                }

                assert_eq!(Matchups::new(&ss).collect::<Vec<_>>(), expected);
                ss.simulate_round(&mut rng);
            }
        }
    }

    fn even_field() -> SwissSystem {
        SwissSystem::new([Rating::new(1000); 16], Sigma::default())
    }

    /// Use the rulebook's one-based initial seed numbers in draw fixtures.
    fn draw_by_seed(ss: &SwissSystem) -> Vec<(u16, u16)> {
        Matchups::new(ss)
            .map(|(a, b)| (a.to_u16() + 1, b.to_u16() + 1))
            .collect()
    }

    /// Reference draw for rounds two and three, independent of the lookup tables.
    fn high_low_draw(ss: &SwissSystem) -> Vec<Pair> {
        // Keep the highest seed's lowest available opponent unless doing so
        // would force a rematch elsewhere in the group.
        fn pair_group(ss: &SwissSystem, group: &[Index]) -> Option<Vec<Pair>> {
            let Some((&highest, others)) = group.split_first() else {
                return Some(Vec::new());
            };

            for &lowest in others.iter().rev() {
                if ss.opponents(highest).contains(lowest) {
                    continue;
                }

                let rest = others
                    .iter()
                    .copied()
                    .filter(|&t| t != lowest)
                    .collect::<Vec<_>>();

                if let Some(mut pairs) = pair_group(ss, &rest) {
                    pairs.insert(0, (highest, lowest));
                    return Some(pairs);
                }
            }

            None
        }

        let mut draw = Vec::new();

        // Everyone has played the same number of matches, so wins identify the record.
        for wins in (0..=ss.rounds_complete()).rev() {
            let mut group: Vec<_> = Index::iter_all().filter(|&t| ss.wins(t) == wins).collect();

            group.sort_by_cached_key(|&team| {
                let buchholz: i16 = ss
                    .opponents(team)
                    .into_iter()
                    .map(|opp| i16::from(ss.wins(opp)) - i16::from(ss.losses(opp)))
                    .sum();
                (Reverse(buchholz), team)
            });

            draw.extend(pair_group(ss, &group).expect("no rematch-free draw"));
        }

        draw
    }

    fn assert_scripted_draw(results: &[u8], expected: &[(u16, u16)]) {
        let mut ss = even_field();

        for &round in results {
            ss.simulate_round(&mut rng::ScriptedMatchRng(round));
        }

        assert_eq!(draw_by_seed(&ss), expected, "round results: {results:02x?}");
    }

    /// Checks all eight pairings in Valve's [Initial Swiss Match-ups] table:
    /// each initial seed from 1 through 8 plays the seed eight places below it.
    ///
    /// [Initial Swiss Match-ups]: https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#initial-swiss-match-ups
    #[test]
    fn first_round_matchups() {
        // Ensure that first round pairs place the best seed as the first team, otherwise
        // SECOND_ROUND_LUT will break
        for (a, b) in FIRST_ROUND {
            assert!(a < b);
        }

        let expected = (1..=8).map(|seed| (seed, seed + 8)).collect::<Vec<_>>();
        assert_eq!(draw_by_seed(&even_field()), expected);
    }

    /// Checks the [Swiss Bracket] draw for all 256 opening-round outcomes. Within
    /// each record group (1–0 or 0–1), Buchholz scores are tied, so the highest
    /// remaining initial seed plays the lowest.
    ///
    /// [Swiss Bracket]: https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket
    #[test]
    fn second_round_matchups() {
        let fresh = even_field();

        for first in 0..=u8::MAX {
            let mut after_first = fresh;
            after_first.simulate_round(&mut rng::ScriptedMatchRng(first));

            assert_eq!(
                Matchups::new(&after_first).collect::<Vec<_>>(),
                high_low_draw(&after_first),
                "round one results: {first:08b}"
            );
        }
    }

    /// Checks the [Swiss Bracket] draw after all 65,536 possible two-round histories.
    /// Within each record group, [Mid-stage Seed Calculation] breaks ties by Buchholz
    /// score, then initial seed. The highest remaining seed plays the lowest seed
    /// that permits a draw without rematches.
    ///
    /// [Swiss Bracket]: https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket
    /// [Mid-stage Seed Calculation]: https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#mid-stage-seed-calculation
    #[test]
    fn third_round_matchups() {
        let fresh = even_field();

        for first in 0..=u8::MAX {
            let mut after_first = fresh;
            after_first.simulate_round(&mut rng::ScriptedMatchRng(first));

            for second in 0..=u8::MAX {
                let mut after_second = after_first;
                after_second.simulate_round(&mut rng::ScriptedMatchRng(second));

                assert_eq!(
                    Matchups::new(&after_second).collect::<Vec<_>>(),
                    high_low_draw(&after_second),
                    "round one/two results: {first:08b}/{second:08b}"
                );
            }
        }
    }

    /// Checks rows 1 through 5 of the [Swiss Bracket] priority table for both
    /// round-four groups, including draws where only one group must skip a row
    /// that is legal for the other.
    ///
    /// [Swiss Bracket]: https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket
    #[test]
    fn fourth_round_matchups() {
        let fixtures = [
            (
                [0xff; 3],
                [(3, 8), (4, 7), (5, 6), (9, 14), (10, 13), (11, 12)],
            ),
            (
                [0xd8, 0x6e, 0x6a],
                [(7, 3), (8, 2), (10, 9), (11, 12), (4, 16), (6, 15)],
            ),
            (
                [0xff, 0x01, 0xaa],
                [(7, 14), (5, 15), (2, 4), (8, 12), (3, 16), (13, 10)],
            ),
            (
                [0x00, 0x60, 0x04],
                [(10, 12), (13, 9), (16, 2), (11, 4), (5, 1), (8, 3)],
            ),
            (
                [0xaa, 0xaa, 0x01],
                [(8, 13), (15, 16), (12, 9), (3, 10), (6, 7), (2, 14)],
            ),
        ];

        for (results, expected) in fixtures {
            assert_scripted_draw(&results, &expected);
        }
    }

    /// Checks round-five draws where the first legal [Swiss Bracket] priority
    /// row ranges from 1 to 12, including draws that must pair two top-three
    /// seeds to avoid a rematch.
    ///
    /// [Swiss Bracket]: https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket
    #[test]
    fn fifth_round_matchups() {
        let fixtures = [
            ([0xff, 0x01, 0xaa, 0x01], [(5, 16), (2, 12), (14, 10)]),
            ([0xff, 0x01, 0xaa, 0x00], [(7, 16), (5, 10), (2, 12)]),
            ([0xff, 0x01, 0xaa, 0x03], [(2, 12), (15, 16), (14, 10)]),
            ([0xff, 0x01, 0xaa, 0x0a], [(7, 10), (2, 8), (15, 16)]),
            ([0xff, 0x01, 0xaa, 0x0b], [(2, 14), (15, 16), (8, 10)]),
            ([0xff, 0x01, 0xaa, 0x02], [(7, 10), (2, 12), (15, 16)]),
            ([0x70, 0xbe, 0x57, 0x2a], [(5, 15), (7, 1), (11, 4)]),
            ([0x13, 0x9a, 0xab, 0x13], [(16, 9), (7, 12), (1, 8)]),
            ([0x05, 0x80, 0x67, 0x32], [(1, 3), (8, 11), (6, 9)]),
            ([0xff, 0xbc, 0x8e, 0x0d], [(7, 6), (10, 14), (2, 15)]),
            ([0xe7, 0x67, 0x3c, 0x3f], [(8, 9), (13, 12), (16, 5)]),
            ([0x0f, 0x43, 0x2d, 0x0d], [(4, 14), (3, 15), (11, 12)]),
        ];

        for (results, expected) in fixtures {
            assert_scripted_draw(&results, &expected);
        }
    }

    /// Checks the [Swiss Bracket] rules across all five rounds of simulated stages:
    /// each active team plays once per round against a new opponent with the same record.
    /// Teams that reach three wins or three losses receive no further matches in the stage.
    ///
    /// [Swiss Bracket]: https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md#swiss-bracket
    #[test]
    fn match_same_group_without_rematches() {
        let fresh = even_field();
        let mut rng = rng::deterministic();

        for _ in 0..64 {
            let mut ss = fresh;

            for _ in 1..=5 {
                let eligible: Set = Index::iter_all()
                    .filter(|&t| ss.wins(t) < 3 && ss.losses(t) < 3)
                    .collect();
                let mut scheduled = Set::new();

                for (a, b) in Matchups::new(&ss) {
                    assert_eq!((ss.wins(a), ss.losses(a)), (ss.wins(b), ss.losses(b)));
                    assert!(!ss.opponents(a).contains(b), "rematch: {a} vs {b}");
                    assert!(scheduled.insert(a) && scheduled.insert(b), "duplicate team");
                }

                assert_eq!(scheduled, eligible);
                ss.simulate_round(&mut rng);
            }
        }
    }
}
