use std::{iter::Sum, ops::Add};

use crate::simulation::{Simulation, SwissSystem};

mod assess;
mod basic;
mod picks;
mod strength;

pub use assess::AssessReport;
pub use basic::BasicReport;
pub use picks::PicksReport;
pub use strength::StrengthReport;

/// Interface for aggregating completed tournament iterations into displayable output.
pub trait Report: Copy + Send + Sum + Sync {
    /// Update this report with one completed Swiss-system tournament.
    fn update(&mut self, ss: &SwissSystem);
    /// Format the aggregated report for the command-line interface.
    fn format(&self, sim: &Simulation) -> String;
}

/// Report that composes the main report types into one output.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReportAll {
    pub basic: BasicReport,
    pub strength: StrengthReport,
    pub picks: PicksReport,
}

impl Add for ReportAll {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.basic = self.basic + rhs.basic;
        self.strength = self.strength + rhs.strength;
        self.picks = self.picks + rhs.picks;
        self
    }
}

impl Sum for ReportAll {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, report| acc + report)
    }
}

impl Report for ReportAll {
    fn update(&mut self, ss: &SwissSystem) {
        self.basic.update(ss);
        self.strength.update(ss);
        self.picks.update(ss);
    }

    fn format(&self, sim: &Simulation) -> String {
        format!(
            "{}\n{}\n{}",
            self.picks.format(sim),
            self.basic.format(sim),
            self.strength.format(sim)
        )
    }
}

/// Report type to use for benchmarking without optimising away simulation work.
#[derive(Debug, Clone, Copy)]
pub struct NullReport;

impl Sum for NullReport {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self, |_, report| std::hint::black_box(report))
    }
}

impl Report for NullReport {
    fn update(&mut self, _ss: &SwissSystem) {
        *self = std::hint::black_box(Self);
    }

    fn format(&self, _: &Simulation) -> String {
        format!("{self:?}")
    }
}
