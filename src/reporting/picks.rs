use std::{
    collections::{BinaryHeap, HashSet},
    iter::Sum,
    ops::Add,
};

use crate::{
    datatypes::Index,
    reporting::{AssessReport, BasicReport, Report},
    simulation::{Simulation, SwissSystem},
};

/// Candidate team for one pick category.
///
/// Equality and hashing intentionally use only the team index so a candidate can
/// move between probability-ranked pools without becoming a distinct set entry.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    index: Index,
    probability: f32,
}

impl std::hash::Hash for Candidate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
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

/// Build a max-heap for a pick category so the highest probability candidate is
/// always popped first.
fn candidates(
    probabilities: [f32; 16],
    exclude: &[&HashSet<Candidate>],
) -> impl Iterator<Item = Candidate> {
    probabilities
        .into_iter()
        .enumerate()
        .map(|(i, probability)| Candidate {
            index: unsafe { Index::from_usize(i) },
            probability,
        })
        .filter(|candidate| exclude.iter().all(|set| !set.contains(candidate)))
}

/// Report for selecting pick recommendations from basic outcome probabilities.
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
        let [tz, adv, zt] = self.basic.calculate_probabilities(sim);

        // Start with the best advancement picks.
        let mut adv_candidates = candidates(adv, &[]).collect::<Vec<_>>();
        adv_candidates.sort_unstable_by(|a, b| b.cmp(a));
        let mut adv_picks = adv_candidates.into_iter().take(6).collect::<HashSet<_>>();

        // Optimise 3-0 picks by swapping previous advancement picks to maximise win probability.
        let mut tz_picks = HashSet::new();
        let mut tz_candidates = BinaryHeap::new();
        let mut swap_candidates = BinaryHeap::new();

        // Populate the pool of swap candidates with 3-0 candidates that are also
        // picked for advancement.
        for candidate in candidates(tz, &[]) {
            if adv_picks.contains(&candidate) {
                swap_candidates.push(candidate);
            } else {
                tz_candidates.push(candidate);
            }
        }

        // Populate the pool of advancement candidates with unpicked candidates.
        let mut adv_candidates = candidates(adv, &[&adv_picks]).collect::<BinaryHeap<_>>();

        while tz_picks.len() < 2 {
            let next_tz = tz_candidates.pop();
            let next_swap_tz = swap_candidates.pop();
            let next_swap_adv = next_swap_tz.and_then(|next| adv_picks.get(&next));
            let mut next_adv = adv_candidates.pop();

            // Ensure that the next advancement pick hasn't already been picked for 3-0.
            while next_adv.is_some_and(|next| tz_picks.contains(&next)) {
                next_adv = adv_candidates.pop();
            }

            match (next_tz, next_swap_tz, next_swap_adv, next_adv) {
                // There are teams in all relevant candidate pools.
                (Some(next_tz), Some(next_swap_tz), Some(next_swap_adv), Some(next_adv)) => {
                    // Compare the lost advancement probability against the
                    // gained 3-0 probability from making the swap.
                    let cost = next_adv.probability - next_swap_adv.probability;
                    let reward = next_swap_tz.probability - next_tz.probability;

                    // Repopulate candidate pools with candidates that remain
                    // unselected so the next loop considers them again.
                    if reward > cost {
                        tz_picks.insert(next_swap_tz);
                        adv_picks.remove(&next_swap_tz);
                        adv_picks.insert(next_adv);
                        tz_candidates.push(next_tz);
                    } else {
                        tz_picks.insert(next_tz);
                        adv_candidates.push(next_adv);
                        swap_candidates.push(next_swap_tz);
                    }
                }
                // No viable swaps left, fill picks straight from the 3-0 pool.
                (Some(next_tz), None, ..) => _ = tz_picks.insert(next_tz),
                // The current state no longer makes any sense, either the 3-0 pool is empty
                // or the advancement picks don't contain the next swap candidate.
                state => unreachable!("invalid state for picking 3-0 teams:\n\n{state:#?}"),
            }
        }

        // Choose 0-3 picks.
        let mut zt_candidates = candidates(zt, &[&tz_picks, &adv_picks]).collect::<Vec<_>>();
        zt_candidates.sort_unstable_by(|a, b| b.cmp(a));
        let zt_picks = zt_candidates.into_iter().take(2).collect::<HashSet<_>>();

        // The picks at this point are potentially still suboptimal, in future I want to further optimise picks
        // using A* or similar to explore more combinations.

        // Assess picks through a second simulation pass using the chosen teams.
        let assessment = sim.run(AssessReport::new(
            tz_picks.iter().map(|c| c.index),
            adv_picks.iter().map(|c| c.index),
            zt_picks.iter().map(|c| c.index),
        ));

        // Format results into a string.
        let mut out = Vec::with_capacity(15);

        for (title, picks) in [
            ("\n3-0 picks:", tz_picks),
            ("\n3-1 or 3-2 picks:", adv_picks),
            ("\n0-3 picks:", zt_picks),
        ] {
            out.push(String::from(title));
            let mut picks = picks.into_iter().collect::<Vec<_>>();
            picks.sort_by(|a, b| b.probability.total_cmp(&a.probability));

            for (i, pick) in picks.into_iter().enumerate() {
                out.push(format!(
                    "{num:<4}{name:<20}{p:>6.1}%",
                    num = format!("{}.", i + 1),
                    name = sim.teams.names[pick.index.to_usize()],
                    p = pick.probability * 100.0
                ));
            }
        }

        out.push(assessment.format(sim));
        out.join("\n")
    }
}
