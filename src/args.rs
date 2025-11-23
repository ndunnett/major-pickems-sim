use std::path::PathBuf;

use clap::{Arg, Command, ValueEnum, builder::PossibleValue, value_parser};
use itertools::Itertools;

#[derive(Clone, Copy, PartialEq)]
pub(super) enum ReportType {
    All,
    Basic,
    Strength,
    Picks,
    Assess,
}

impl ValueEnum for ReportType {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            ReportType::All,
            ReportType::Basic,
            ReportType::Strength,
            ReportType::Picks,
            ReportType::Assess,
        ]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            ReportType::All => PossibleValue::new("all").help("includes all statistics"),
            ReportType::Basic => PossibleValue::new("basic")
                .help("3-0, advancment, and 0-3 percentages for each team"),
            ReportType::Strength => PossibleValue::new("strength")
                .help("relative strength of opponents faced for each team"),
            ReportType::Picks => PossibleValue::new("picks").help("statistically optimal picks"),
            ReportType::Assess => PossibleValue::new("assess").help("simulated outcome of picks"),
        })
    }
}

#[derive(Clone)]
pub(super) enum ExtraArgs {
    Assess {
        three_zero: [String; 2],
        advancing: [String; 6],
        zero_three: [String; 2],
    },
}

pub(super) enum Args {
    Simulate {
        file: PathBuf,
        sigma: f32,
        iterations: u64,
        report: ReportType,
        extra_args: Option<Box<ExtraArgs>>,
    },
    Inspect {
        file: PathBuf,
    },
    Wizard {
        file: PathBuf,
    },
}

impl Args {
    fn cmd() -> Command {
        Command::new("pickems")
            .version(env!("CARGO_PKG_VERSION"))
            .about(env!("CARGO_PKG_DESCRIPTION"))
            .subcommand_required(true)
            .arg_required_else_help(true)
            .subcommand(
                Command::new("simulate")
                    .about("Simulate tournament outcomes")
                    .arg(
                        Arg::new("file")
                            .short('f')
                            .long("file")
                            .help("Path to load input data from (.toml)")
                            .required(true)
                            .value_parser(value_parser!(PathBuf)),
                    )
                    .arg(
                        Arg::new("iterations")
                            .short('n')
                            .long("iterations")
                            .default_value("1000000")
                            .help("Number of iterations to run")
                            .value_parser(value_parser!(u64)),
                    )
                    .arg(
                        Arg::new("sigma")
                            .short('s')
                            .long("sigma")
                            .default_value("800.0")
                            .help("Sigma value to use for win probability")
                            .value_parser(value_parser!(f32)),
                    )
                    .arg(
                        Arg::new("report")
                            .short('r')
                            .long("report")
                            .default_value("basic")
                            .requires_ifs([
                                ("assess", "three-zero"),
                                ("assess", "advancing"),
                                ("assess", "zero-three"),
                            ])
                            .help("Report format to return")
                            .value_parser(value_parser!(ReportType)),
                    )
                    .arg(
                        Arg::new("three-zero")
                            .long("three-zero")
                            .required_if_eq("report", "assess")
                            .num_args(2)
                            .help("3-0 picks")
                            .value_parser(value_parser!(String)),
                    )
                    .arg(
                        Arg::new("advancing")
                            .long("advancing")
                            .required_if_eq("report", "assess")
                            .num_args(6)
                            .help("3-1/3-2 picks")
                            .value_parser(value_parser!(String)),
                    )
                    .arg(
                        Arg::new("zero-three")
                            .long("zero-three")
                            .required_if_eq("report", "assess")
                            .num_args(2)
                            .help("0-3 picks")
                            .value_parser(value_parser!(String)),
                    ),
            )
            .subcommand(
                Command::new("data")
                    .about("Inspect or create input data")
                    .subcommand_required(true)
                    .arg_required_else_help(true)
                    .subcommand(
                        Command::new("inspect")
                            .about("Inspect input data file")
                            .arg(
                                Arg::new("file")
                                    .short('f')
                                    .long("file")
                                    .help("Path to load input data from (.toml)")
                                    .required(true)
                                    .value_parser(value_parser!(PathBuf)),
                            ),
                    )
                    .subcommand(
                        Command::new("wizard")
                            .about("Use the data wizard to create an input data file")
                            .arg(
                                Arg::new("file")
                                    .short('f')
                                    .long("file")
                                    .help("Path to save input data to (.toml)")
                                    .required(true)
                                    .value_parser(value_parser!(PathBuf)),
                            ),
                    ),
            )
    }

    pub fn parse() -> Option<Self> {
        let matches = Self::cmd().get_matches();

        if let Some(sim) = matches.subcommand_matches("simulate") {
            let report = *sim.get_one::<ReportType>("report")?;

            let extra_args = if report == ReportType::Assess {
                Some(Box::new(ExtraArgs::Assess {
                    three_zero: sim.get_many("three-zero")?.cloned().collect_array()?,
                    advancing: sim.get_many("advancing")?.cloned().collect_array()?,
                    zero_three: sim.get_many("zero-three")?.cloned().collect_array()?,
                }))
            } else {
                None
            };

            return Some(Self::Simulate {
                file: sim.get_one::<PathBuf>("file")?.clone(),
                sigma: *sim.get_one::<f32>("sigma")?,
                iterations: *sim.get_one::<u64>("iterations")?,
                report,
                extra_args,
            });
        } else if let Some(data) = matches.subcommand_matches("data") {
            if let Some(inspect) = data.subcommand_matches("inspect") {
                return Some(Self::Inspect {
                    file: inspect.get_one::<PathBuf>("file")?.clone(),
                });
            } else if let Some(wizard) = data.subcommand_matches("wizard") {
                return Some(Self::Wizard {
                    file: wizard.get_one::<PathBuf>("file")?.clone(),
                });
            }
        }

        None
    }
}
