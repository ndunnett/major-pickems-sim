#![allow(clippy::derive_partial_eq_without_eq)]

use std::{path::PathBuf, str::FromStr};

use argh::{FromArgValue, FromArgs};
use pickems::datatypes::{Iterations, Name, Sigma};

#[derive(FromArgs, PartialEq, Debug)]
/// Simulate tournament stage outcomes for Counter-Strike major tournaments.
pub struct Args {
    #[argh(subcommand)]
    pub command: Command,
}

#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand)]
pub enum Command {
    Simulate(Simulate),
    Update(Update),
    Inspect(Inspect),
    Tui(Tui),
}

#[derive(Debug, PartialEq, FromArgValue)]
pub enum ReportSelection {
    All,
    Basic,
    Strength,
    Picks,
    Assess,
}

fn parse_sigma(s: &str) -> Result<Sigma, String> {
    Sigma::from_str(s).map_err(|e| e.to_string())
}

fn parse_iterations(s: &str) -> Result<Iterations, String> {
    Iterations::from_str(s).map_err(|e| e.to_string())
}

fn parse_names<const N: usize>(s: &str) -> Result<Box<[Name; N]>, String> {
    let validated_names = s
        .split(',')
        .map(Name::validate)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    let len = validated_names.len();

    if len == N {
        Ok(Box::new(std::array::from_fn(|i| unsafe {
            Name::new_unchecked(String::from(validated_names[i]))
        })))
    } else {
        Err(format!("must declare exactly {N} teams"))
    }
}

#[derive(FromArgs, PartialEq, Debug)]
/// Run simulation and print report.
#[argh(subcommand, name = "simulate")]
pub struct Simulate {
    /// path to the input data file
    #[argh(option, short = 'f')]
    pub file: PathBuf,
    /// number of iterations to run [default: 1000000]
    #[argh(option, short = 'n', from_str_fn(parse_iterations))]
    pub iterations: Iterations,
    /// sigma value to use for win probability [default: 800]
    #[argh(option, short = 's', from_str_fn(parse_sigma))]
    pub sigma: Sigma,
    /// report data to collect [default: basic]
    #[argh(option, short = 'r', default = "ReportSelection::Basic")]
    pub report: ReportSelection,
    /// three-zero team picks for assess report
    #[argh(option, from_str_fn(parse_names::<2>))]
    pub three_zero: Option<Box<[Name; 2]>>,
    /// advance team picks for assess report
    #[argh(option, from_str_fn(parse_names::<6>))]
    pub advance: Option<Box<[Name; 6]>>,
    /// zero-three team picks for assess report
    #[argh(option, from_str_fn(parse_names::<2>))]
    pub zero_three: Option<Box<[Name; 2]>>,
}

fn default_data_path() -> PathBuf {
    PathBuf::from("./data")
}

#[derive(FromArgs, PartialEq, Debug)]
/// Update input data from remote repository.
#[argh(subcommand, name = "update")]
pub struct Update {
    #[argh(option, short = 'p', default = "default_data_path()")]
    /// path to the local data directory [default: "./data"]
    pub path: PathBuf,
}

#[derive(FromArgs, PartialEq, Debug)]
/// Print input data file.
#[argh(subcommand, name = "inspect")]
pub struct Inspect {
    /// path to the input data file
    #[argh(option, short = 'f')]
    pub file: PathBuf,
}

#[derive(FromArgs, PartialEq, Debug)]
/// Launch the interactive terminal interface.
#[argh(subcommand, name = "tui")]
pub struct Tui {
    #[argh(option, short = 'p', default = "default_data_path()")]
    /// path to the local data directory [default: "./data"]
    pub path: PathBuf,
}
