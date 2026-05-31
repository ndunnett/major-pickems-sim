use std::time::Instant;

pub mod args;
pub mod tui;
pub mod update;

use args::{Args, Command, ReportSelection};
use pickems::{
    datatypes::{Iterations, Sigma, Teams},
    reporting::{AssessReport, BasicReport, PicksReport, Report, ReportAll, StrengthReport},
    simulation::Simulation,
};

fn run_and_format(
    teams: Teams,
    sigma: Sigma,
    iterations: Iterations,
    report: impl Report,
) -> String {
    let sim = Simulation::new(teams, sigma, iterations);
    let report = sim.run(report);
    report.format(&sim)
}

pub fn run() -> anyhow::Result<()> {
    let args: Args = argh::from_env();

    match args.command {
        Command::Simulate(simulate) => {
            let now = Instant::now();
            let teams = Teams::parse_toml(simulate.file)?;
            let s = simulate.sigma;
            let n = simulate.iterations;

            let formatted_report = match simulate.report {
                ReportSelection::All => run_and_format(teams, s, n, ReportAll::default()),
                ReportSelection::Basic => run_and_format(teams, s, n, BasicReport::default()),
                ReportSelection::Strength => run_and_format(teams, s, n, StrengthReport::default()),
                ReportSelection::Picks => run_and_format(teams, s, n, PicksReport::default()),
                ReportSelection::Assess => {
                    let Some(tz) = simulate.three_zero else {
                        anyhow::bail!("the \"assess\" report requires the --three-zero argument");
                    };

                    let Some(adv) = simulate.advance else {
                        anyhow::bail!("the \"assess\" report requires the --advance argument");
                    };

                    let Some(zt) = simulate.zero_three else {
                        anyhow::bail!("the \"assess\" report requires the --zero-three argument");
                    };

                    let report = AssessReport::try_from_args(&teams, &tz, &adv, &zt)?;
                    run_and_format(teams, s, n, report)
                }
            };

            // Format number of iterations into a string, with thousands separated by commas.
            let formatted_n = n
                .to_string()
                .as_bytes()
                .rchunks(3)
                .rev()
                .map(str::from_utf8)
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
                .join(",");

            let seconds = now.elapsed().as_millis() as f32 / 1000.0;

            println!(
                "RESULTS FROM {formatted_n} TOURNAMENT SIMULATIONS\n{formatted_report}\n\nRun time: {seconds} seconds"
            );

            Ok(())
        }
        Command::Update(update) => {
            let mut updated = false;

            for name in update::data_updater(&update.path)? {
                println!("Downloaded {}", name?);
                updated = true;
            }

            if !updated {
                println!("No updates available.");
            }

            Ok(())
        }
        Command::Inspect(inspect) => {
            let teams = pickems::datatypes::Teams::parse_toml(inspect.file)?;
            let mut out = Vec::new();
            out.push(format!("{:<4}  {:<18}{:>6}", "Seed", "Team", "Rating"));

            for (seed, (name, rating)) in teams.names.iter().zip(teams.ratings.iter()).enumerate() {
                out.push(format!(
                    "{:<4}  {:<18}{:>6}",
                    format!("{}.", seed + 1),
                    name,
                    rating
                ));
            }

            println!("{}", out.join("\n"));
            Ok(())
        }
        Command::Tui(tui) => tui::run(tui.path),
    }
}
