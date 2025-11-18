#![feature(portable_simd)]
#![cfg_attr(
    feature = "pprof",
    allow(dead_code, unreachable_code, unused_imports, unused_variables)
)]

mod data;
mod simulate;
mod wizard;

pub use data::{inspect, wizard};
pub use simulate::{BasicReport, NullReport, Simulation, simulate};
