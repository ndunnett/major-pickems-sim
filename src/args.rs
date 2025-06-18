use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Path to input data (.json)
    #[arg(short, long)]
    pub file: PathBuf,

    /// Number of iterations to run
    #[arg(short = 'n', long, default_value_t = 1_000_000)]
    pub iterations: u32,

    /// Sigma value to use for win probability
    #[arg(short, long, default_value_t = 800.0)]
    pub sigma: f64,
}
