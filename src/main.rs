#![cfg_attr(
    feature = "pprof",
    allow(dead_code, unreachable_code, unused_imports, unused_variables)
)]

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

mod data;
mod simulate;
mod wizard;

use crate::{
    data::{inspect, wizard},
    simulate::simulate,
};

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Simulate tournament outcomes
    Simulate {
        #[command(flatten)]
        args: SimulateArgs,
    },
    /// Inspect or create input data
    Data {
        #[command(subcommand)]
        subcommand: DataSubcommand,
    },
}

#[derive(Debug, Args)]
pub struct SimulateArgs {
    /// Path to load input data from (.toml)
    #[arg(short, long)]
    file: PathBuf,

    /// Number of iterations to run
    #[arg(short = 'n', long, default_value_t = 1_000_000)]
    iterations: u64,

    /// Sigma value to use for win probability
    #[arg(short, long, default_value_t = 800.0)]
    sigma: f64,
}

#[derive(Debug, Subcommand)]
pub enum DataSubcommand {
    /// Inspect input data file
    Inspect {
        /// Path to load input data from (.toml)
        #[arg(short, long)]
        file: PathBuf,
    },
    /// Use the data wizard to create an input data file
    Wizard {
        /// Path to save input data to (.toml)
        #[arg(short, long)]
        file: PathBuf,
    },
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Simulate { args } => simulate(args.file, args.iterations, args.sigma)?,
        Command::Data { subcommand } => match subcommand {
            DataSubcommand::Inspect { file } => inspect(file)?,
            DataSubcommand::Wizard { file } => wizard(file)?,
        },
    }

    Ok(())
}

#[cfg(not(feature = "pprof"))]
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
    }
}

#[cfg(feature = "pprof")]
fn main() {
    use pprof::protos::Message;
    use std::{fs::File, io::Write};

    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(10000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();

    crate::simulate::Simulation::bench_test();

    if let Ok(report) = guard.report().build() {
        let mut content = Vec::new();
        let profile = report.pprof().unwrap();
        profile.encode(&mut content).unwrap();
        let mut file = File::create("profile.pb").unwrap();
        file.write_all(&content).unwrap();
    };
}
