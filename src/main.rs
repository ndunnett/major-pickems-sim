use anyhow::anyhow;

use pickems::*;

mod args;

use crate::args::{Args, ExtraArgs, ReportType};

#[cfg(not(feature = "pprof"))]
fn main() -> anyhow::Result<()> {
    match Args::parse() {
        Some(Args::Simulate {
            file,
            sigma,
            iterations,
            report,
            extra_args,
        }) => match report {
            ReportType::All => simulate(file, sigma, iterations, ReportAll::default()),
            ReportType::Basic => simulate(file, sigma, iterations, BasicReport::default()),
            ReportType::Strength => simulate(file, sigma, iterations, StrengthReport::default()),
            ReportType::Picks => simulate(file, sigma, iterations, PicksReport::default()),
            ReportType::Assess => {
                if let Some(ExtraArgs::Assess {
                    three_zero,
                    advancing,
                    zero_three,
                }) = extra_args.as_deref()
                {
                    simulate(
                        file.clone(),
                        sigma,
                        iterations,
                        AssessReport::try_from_args(file, three_zero, advancing, zero_three)?,
                    )
                } else {
                    Err(anyhow!("failed to parse args"))
                }
            }
        },
        Some(Args::Inspect { file }) => inspect(file),
        Some(Args::Wizard { file }) => wizard(file),
        None => Err(anyhow!("failed to parse CLI arguments")),
    }
}

#[cfg(feature = "pprof")]
fn main() {
    use pprof::protos::Message;
    use std::{fs::File, io::Write, time::Instant};

    let now = Instant::now();
    let guard = pprof::ProfilerGuard::new(1000).unwrap();
    _ = Simulation::bench_test(1000000);

    println!(
        "Run time: {} seconds",
        now.elapsed().as_millis() as f32 / 1000.0
    );

    if let Ok(report) = guard.report().build() {
        let mut content = Vec::new();
        let profile = report.pprof().unwrap();
        profile.encode(&mut content).unwrap();
        let mut file = File::create("target/profile.pb").unwrap();
        file.write_all(&content).unwrap();
    };
}
