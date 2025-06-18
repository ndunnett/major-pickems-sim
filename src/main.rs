use std::time::Duration;

use clap::Parser;
use itertools::Itertools;

mod args;
mod simulate;

use crate::{
    args::Args,
    simulate::{Simulation, TeamIndex, TeamResult},
};

type TeamResultField = fn(&TeamResult) -> u64;

/// Format results from a simulation into a readable/printable string.
fn format_results(results: TeamIndex<TeamResult>, iterations: u32, run_time: Duration) -> String {
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

    let fields: [(TeamResultField, &str); 3] = [
        (|result| result.three_zero, "3-0"),
        (|result| result.advanced, "3-1 or 3-2"),
        (|result| result.zero_three, "0-3"),
    ];

    for (func, title) in fields.iter() {
        out.push(format!("\nMost likely to {title}:"));

        let sorted_results = results
            .items()
            .sorted_by(|(_, a), (_, b)| func(b).cmp(&func(a)))
            .enumerate();

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

fn main() {
    let args = Args::parse();
    let now = std::time::Instant::now();
    let sim = Simulation::try_from_file(args.file, args.sigma).unwrap();
    let results_index = sim.run(args.iterations);

    println!(
        "{}",
        format_results(results_index, args.iterations, now.elapsed())
    );
}
