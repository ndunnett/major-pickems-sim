use std::time::Instant;

use pprof::protos::Message;

use pickems::{datatypes::Iterations, reporting::NullReport, simulation::Simulation};

const BATCHES: u64 = 500;
const ITERATIONS: u64 = 1_000_000;

pub fn run() {
    let sim = Simulation::dummy(Iterations::new(ITERATIONS));
    let now = Instant::now();

    let report = {
        let guard = pprof::ProfilerGuard::new(10000).unwrap();

        for _ in 0..BATCHES {
            sim.bench_test(NullReport);
        }

        guard.report().build().expect("could not build CPU profile")
    };

    println!(
        "{:.0}M tournaments in {:.3}s",
        BATCHES as f32 * ITERATIONS as f32 / 1_000_000.0,
        now.elapsed().as_secs_f64()
    );

    let profile = report.pprof().unwrap().encode_to_vec();
    std::fs::write("target/profile.pb", profile).unwrap();
}
