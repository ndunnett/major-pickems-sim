use std::path::PathBuf;

use clap::{Arg, Command, ValueEnum, builder::PossibleValue, value_parser};

#[derive(Clone, Copy)]
pub(super) enum ReportType {
    All,
    Basic,
    Strength,
    Picks,
}

impl ValueEnum for ReportType {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            ReportType::All,
            ReportType::Basic,
            ReportType::Strength,
            ReportType::Picks,
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
        })
    }
}

pub(super) enum Args {
    Simulate {
        file: PathBuf,
        sigma: f32,
        iterations: u64,
        report: ReportType,
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
                            .help("Report format to return")
                            .value_parser(value_parser!(ReportType)),
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
            return Some(Self::Simulate {
                file: sim.get_one::<PathBuf>("file")?.clone(),
                sigma: *sim.get_one::<f32>("sigma")?,
                iterations: *sim.get_one::<u64>("iterations")?,
                report: *sim.get_one::<ReportType>("report")?,
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
