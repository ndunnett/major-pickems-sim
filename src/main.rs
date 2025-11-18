use anyhow::anyhow;

mod args;
use args::Args;

use pickems::*;

#[cfg(not(feature = "pprof"))]
fn main() -> anyhow::Result<()> {
    match Args::parse() {
        Some(Args::Simulate {
            file,
            iterations,
            sigma,
        }) => simulate(file, iterations, sigma),
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
