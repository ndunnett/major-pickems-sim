use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

mod data;
mod simulate;

use crate::simulate::simulate;

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

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Simulate { args } => simulate(args.file, args.iterations, args.sigma)?,
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
    }
}
