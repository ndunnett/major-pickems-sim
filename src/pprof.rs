use std::{fs::File, io::Write, time::Instant};

use pprof::protos::Message;

use pickems::{reporting::NullReport, simulation::Simulation};

pub fn run() {
    let now = Instant::now();
    let guard = pprof::ProfilerGuard::new(1000).unwrap();
    _ = Simulation::dummy(1_000_000).bench_test(NullReport);

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
    }
}
