#[cfg(not(feature = "pprof"))]
mod cli;

#[cfg(not(feature = "pprof"))]
fn main() -> anyhow::Result<()> {
    cli::args::Args::parse().map_or_else(
        || Err(anyhow::anyhow!("failed to parse CLI arguments")),
        cli::run,
    )
}

#[cfg(feature = "pprof")]
mod pprof;

#[cfg(feature = "pprof")]
fn main() {
    pprof::run();
}
