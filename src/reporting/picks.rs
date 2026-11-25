use std::{iter::Sum, ops::Add};

use crate::{
    datatypes::{Index, Pool, Set},
    reporting::{AssessReport, BasicReport, Report},
    simulation::{Simulation, SwissSystem},
};

#[derive(Debug, Clone, Copy)]
struct Probabilities {
    data: [[f32; 16]; 3],
}

impl Probabilities {
    fn from_basic(sim: &Simulation, basic: &BasicReport) -> Self {
        Self {
            data: basic.calculate_probabilities(sim),
        }
    }

    const fn tz(&self, index: Index) -> f32 {
        self.data[Pool::ThreeZero as usize][index.to_usize()]
    }

    const fn adv(&self, index: Index) -> f32 {
        self.data[Pool::Advance as usize][index.to_usize()]
    }

    const fn zt(&self, index: Index) -> f32 {
        self.data[Pool::ZeroThree as usize][index.to_usize()]
    }

    fn naive_picks<const N: usize>(&self, pool: Pool, exclude: &[Set]) -> Set {
        let mut candidates = Index::iter_all()
            .filter(|&index| exclude.iter().all(|set| !set.contains(index)))
            .collect::<Vec<_>>();
        let probs = &self.data[pool as usize];
        candidates.sort_unstable_by(|&a, &b| probs[b.to_usize()].total_cmp(&probs[a.to_usize()]));
        candidates.into_iter().take(N).collect::<Set>()
    }

    fn optimal_picks(&self) -> (Set, Set, Set) {
        let mut best_score = 0.0;
        let mut best_picks = None;

        // Iterate all combinations of 3-0 picks with naive advance picks.
        for a in Index::iter_all() {
            for b in Index::iter_all().skip(a.to_usize() + 1) {
                let tz_picks = Set::from_iter([a, b]);
                let adv_picks = self.naive_picks::<6>(Pool::Advance, &[tz_picks]);
                let score = tz_picks.iter().map(|i| self.tz(i)).sum::<f32>()
                    + adv_picks.iter().map(|i| self.adv(i)).sum::<f32>();

                if score > best_score {
                    best_score = score;
                    best_picks = Some((tz_picks, adv_picks));
                }
            }
        }

        let Some((tz_picks, adv_picks)) = best_picks else {
            unreachable!("best_picks will always be Some after iterating all combinations");
        };

        // Choose naive 0-3 picks.
        let zt_picks = self.naive_picks::<2>(Pool::ZeroThree, &[adv_picks, tz_picks]);
        (tz_picks, adv_picks, zt_picks)
    }
}

/// Report for selecting pick recommendations from basic outcome probabilities.
#[derive(Debug, Clone, Default)]
pub struct PicksReport {
    pub basic: BasicReport,
    pub outcomes: Vec<[Set; 3]>,
}

impl PicksReport {
    fn assess(&self, tz_picks: Set, adv_picks: Set, zt_picks: Set) -> AssessReport {
        // Bin tournaments by stars earned (0-10).
        let mut histogram = [0; 11];

        // Each team picked for the category it actually finished in earns one star.
        for &[tz_outcome, adv_outcome, zt_outcome] in &self.outcomes {
            let stars = tz_outcome.intersection(tz_picks).len()
                + adv_outcome.intersection(adv_picks).len()
                + zt_outcome.intersection(zt_picks).len();
            histogram[stars] += 1;
        }

        // Weight each score by its frequency to recover total stars, then divide
        // by the number of tournaments to get mean stars per tournament.
        let mean = histogram
            .iter()
            .enumerate()
            .map(|(stars, &count)| stars as f32 * count as f32)
            .sum::<f32>()
            / self.outcomes.len() as f32;

        // `AssessReport` stores the sum of squared deviations. Each tournament in
        // bin `k` contributes `(k - mean)^2`.
        let ds = histogram
            .iter()
            .enumerate()
            .map(|(stars, &count)| (stars as f32 - mean).powi(2) * count as f32)
            .sum::<f32>();

        // Success count is the sum of the right hand side of the histogram (>= 5 stars).
        let success = histogram[5..].iter().sum();

        AssessReport {
            n: self.outcomes.len() as u64,
            mean,
            ds,
            success,
            three_zero_picks: tz_picks,
            advancing_picks: adv_picks,
            zero_three_picks: zt_picks,
        }
    }
}

impl Add for PicksReport {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.basic = self.basic + rhs.basic;
        self.outcomes.extend(rhs.outcomes);
        self
    }
}

impl Sum for PicksReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, report| acc + report)
    }
}

fn fmt_picks(picks: Set, val: impl Fn(Index) -> f32) -> Vec<(Index, f32)> {
    let mut picks = picks
        .into_iter()
        .map(|index| (index, val(index) * 100.0))
        .collect::<Vec<_>>();
    picks.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    picks
}

impl Report for PicksReport {
    fn update(&mut self, ss: &SwissSystem) {
        self.outcomes.push(self.basic.record(ss));
    }

    fn format(&self, sim: &Simulation) -> String {
        let probs = Probabilities::from_basic(sim, &self.basic);
        let (tz, adv, zt) = probs.optimal_picks();
        let mut out = Vec::with_capacity(15);

        for (title, picks) in [
            ("\n3-0 picks:", fmt_picks(tz, |i| probs.tz(i))),
            ("\n3-1 or 3-2 picks:", fmt_picks(adv, |i| probs.adv(i))),
            ("\n0-3 picks:", fmt_picks(zt, |i| probs.zt(i))),
        ] {
            out.push(String::from(title));

            for (i, (index, probability)) in picks.into_iter().enumerate() {
                out.push(format!(
                    "{num:<4}{name:<20}{probability:>6.1}%",
                    num = format!("{}.", i + 1),
                    name = sim.teams.names[index.to_usize()],
                ));
            }
        }

        out.push(self.assess(tz, adv, zt).format(sim));
        out.join("\n")
    }
}
