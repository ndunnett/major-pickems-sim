use std::{
    collections::{BinaryHeap, HashSet},
    iter::Sum,
    ops::Add,
};

use itertools::Itertools;

use crate::{
    data::TeamSeed,
    simulate::{
        Simulation, SwissSystem,
        reporting::{AssessReport, BasicReport, Report},
    },
};

#[derive(Debug, Clone, Copy)]
struct Candidate {
    seed: TeamSeed,
    probability: f32,
}

impl std::hash::Hash for Candidate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.seed.hash(state);
    }
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.probability.total_cmp(&other.probability)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Report for selecting optimal picks from basic statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct PicksReport {
    pub basic: BasicReport,
}

impl Add for PicksReport {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.basic = self.basic + rhs.basic;
        self
    }
}

impl Sum for PicksReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, report| acc + report)
    }
}

impl Report for PicksReport {
    fn update(&mut self, ss: &SwissSystem) {
        self.basic.update(ss);
    }

    fn format(&self, sim: &Simulation) -> String {
        let [three_zero, advancing, zero_three] = self.basic.calculate_probabilities(sim);

        // Sort out candidates for each category of picks.
        let candidates = |probabilities: [f32; 16]| -> BinaryHeap<Candidate> {
            probabilities
                .iter()
                .enumerate()
                .map(|(i, p)| Candidate {
                    seed: i as TeamSeed,
                    probability: *p,
                })
                .collect::<BinaryHeap<_>>()
        };

        let mut three_zero_candidates = candidates(three_zero);
        let mut advancing_candidates = candidates(advancing);
        let mut zero_three_candidates = candidates(zero_three);

        // Naively select 3-1/3-2 picks.
        let mut advancing_picks = HashSet::new();

        for _ in 0..6 {
            if let Some(candidate) = advancing_candidates.pop() {
                advancing_picks.insert(candidate);
            }
        }

        // Naively select 0-3 picks.
        let mut zero_three_picks = HashSet::new();

        for _ in 0..2 {
            if let Some(candidate) = zero_three_candidates.pop() {
                zero_three_picks.insert(candidate);
            }
        }

        // Optimise 3-0 picks by swapping previous 3-1/3-2 picks to maximise win probability.
        let mut three_zero_picks = HashSet::new();

        while three_zero_picks.len() < 2 {
            let mut swap_candidates = BinaryHeap::new();

            while let Some(candidate) = three_zero_candidates.peek()
                && advancing_picks.contains(candidate)
            {
                swap_candidates.push(three_zero_candidates.pop().unwrap());
            }

            match (
                three_zero_candidates.pop(),
                swap_candidates.pop(),
                advancing_candidates.pop(),
            ) {
                // There are teams in all relevant candidate pools.
                (Some(next_three_zero), Some(next_swap), Some(next_advancing))
                    if advancing_picks.contains(&next_swap) =>
                {
                    let swap_advancing = advancing_picks.get(&next_swap).unwrap();

                    // Calculate delta in win probability for swapping teams.
                    let cost = next_advancing.probability - swap_advancing.probability;
                    let reward = next_swap.probability - next_three_zero.probability;

                    // Swap teams if it is worthwhile, and repopulate candidate pools with candidates that remain unselected.
                    if reward > cost {
                        three_zero_picks.insert(next_swap);
                        advancing_picks.remove(&next_swap);
                        advancing_picks.insert(next_advancing);
                        three_zero_candidates.push(next_three_zero);
                    } else {
                        three_zero_picks.insert(next_three_zero);
                        advancing_candidates.push(next_advancing);
                    }
                }
                // There are only teams left in the 3-0 candidate pool, and unviable candidates in the 3-1/3-2 pool.
                (Some(next_three_zero), None, Some(next_advancing)) => {
                    three_zero_picks.insert(next_three_zero);
                    advancing_candidates.push(next_advancing);
                }
                // There are only teams left in the 3-0 candidate pool.
                (Some(next_three_zero), None, None) => {
                    three_zero_picks.insert(next_three_zero);
                }
                // The current state no longer makes any sense, either the 3-0 pool is empty or the 3-1/3-2 picks don't contain the next swap candidate.
                state => {
                    unreachable!("invalid state for picking 3-0 teams:\n\n{state:#?}");
                }
            }
        }

        // The picks at this point are potentially still suboptimal, in future I want to further optimise picks
        // using A* or similar to explore more combinations.

        // Assess picks through simulation
        let assessment = sim.run(AssessReport::new(
            three_zero_picks.iter().map(|c| c.seed),
            advancing_picks.iter().map(|c| c.seed),
            zero_three_picks.iter().map(|c| c.seed),
        ));

        // Format results into a string
        let format_picks = |out: &mut Vec<String>, picks: &HashSet<Candidate>| {
            for (i, (name, p)) in picks
                .iter()
                .map(|i| (&sim.names[i.seed as usize], i.probability * 100.0))
                .sorted_by(|(_, a), (_, b)| b.total_cmp(a))
                .enumerate()
            {
                out.push(format!(
                    "{num:<4}{name:<20}{p:>6.1}%",
                    num = format!("{}.", i + 1),
                ));
            }
        };

        let mut out = vec![String::from("\n3-0 picks:")];
        format_picks(&mut out, &three_zero_picks);
        out.push(String::from("\n3-1 or 3-2 picks:"));
        format_picks(&mut out, &advancing_picks);
        out.push(String::from("\n0-3 picks:"));
        format_picks(&mut out, &zero_three_picks);
        out.push(assessment.format(sim));
        out.join("\n")
    }
}
