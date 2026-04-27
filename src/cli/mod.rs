use std::{path::PathBuf, time::Instant};

use anyhow::anyhow;
use pickems::{
    datatypes::Teams,
    reporting::{AssessReport, BasicReport, PicksReport, Report, ReportAll, StrengthReport},
    simulation::Simulation,
};

pub mod args;
mod wizard;

use args::{Args, ExtraArgs, ReportType};

pub fn run(args: Args) -> anyhow::Result<()> {
    match args {
        Args::Simulate {
            file,
            sigma,
            iterations,
            report_type,
            extra_args,
        } => run_simulation(file, sigma, iterations, report_type, extra_args.as_deref()),
        Args::Inspect { file } => run_inspect(file),
        Args::Wizard { file } => run_wizard(file),
    }
}

/// Run a tournament simulation and print the report.
fn run_simulation(
    file: PathBuf,
    sigma: f32,
    iterations: u64,
    report_type: ReportType,
    extra_args: Option<&ExtraArgs>,
) -> anyhow::Result<()> {
    let now = Instant::now();
    let teams = Teams::parse_toml(file)?;

    let formatted_report = match report_type {
        ReportType::All => run_and_format(teams, sigma, iterations, ReportAll::default()),
        ReportType::Basic => run_and_format(teams, sigma, iterations, BasicReport::default()),
        ReportType::Strength => run_and_format(teams, sigma, iterations, StrengthReport::default()),
        ReportType::Picks => run_and_format(teams, sigma, iterations, PicksReport::default()),
        ReportType::Assess => {
            if let Some(ExtraArgs::Assess {
                three_zero,
                advancing,
                zero_three,
            }) = extra_args
            {
                let report =
                    AssessReport::try_from_args(&teams, three_zero, advancing, zero_three)?;

                run_and_format(teams, sigma, iterations, report)
            } else {
                return Err(anyhow!("failed to parse args"));
            }
        }
    };

    // Format number of iterations into a string, with thousands separated by commas.
    let formatted_iterations = iterations
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
        "RESULTS FROM {formatted_iterations} TOURNAMENT SIMULATIONS\n{formatted_report}\n\nRun time: {seconds} seconds"
    );

    Ok(())
}

fn run_and_format<R: Report>(teams: Teams, sigma: f32, iterations: u64, report: R) -> String {
    let sim = Simulation::new(teams, sigma, iterations);
    let report = sim.run(report);
    report.format(&sim)
}

/// Print input data loaded from a TOML file.
fn run_inspect(filepath: PathBuf) -> anyhow::Result<()> {
    let teams = Teams::parse_toml(filepath)?;
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

/// Run wizard to generate team data and write to a TOML file.
fn run_wizard(filepath: PathBuf) -> anyhow::Result<()> {
    if let Some(teams) = wizard::Wizard::run()? {
        teams.write_toml(filepath.clone())?;

        println!(
            "Generated input data, saved to '{}':\n",
            filepath.canonicalize()?.display(),
        );

        run_inspect(filepath)?;
    } else {
        println!("Exited wizard without saving.");
    }

    Ok(())
}
