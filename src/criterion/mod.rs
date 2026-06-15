use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

mod full_simulation;
mod probabilities;
mod round_simulation;
mod seeding;
mod sorting;

fn config() -> Criterion {
    Criterion::default()
        .configure_from_args()
        .without_plots()
        .significance_level(0.01)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_millis(500))
}

criterion_group!(
    name =benches;
    config = config();
    targets =
        probabilities::bench,
        sorting::bench,
        seeding::bench,
        round_simulation::bench,
        full_simulation::bench,
);

criterion_main!(benches);
