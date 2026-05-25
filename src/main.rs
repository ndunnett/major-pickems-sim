#[cfg(not(feature = "pprof"))]
mod app;

#[cfg(not(feature = "pprof"))]
fn main() -> anyhow::Result<()> {
    app::run()
}

#[cfg(feature = "pprof")]
mod pprof;

#[cfg(feature = "pprof")]
fn main() {
    pprof::run();
}
