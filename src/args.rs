use std::path::PathBuf;

use clap::{Arg, Command};

pub(super) enum Args {
    Simulate {
        file: PathBuf,
        iterations: u64,
        sigma: f32,
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
                            .value_parser(clap::value_parser!(PathBuf)),
                    )
                    .arg(
                        Arg::new("iterations")
                            .short('n')
                            .long("iterations")
                            .default_value("1000000")
                            .help("Number of iterations to run")
                            .value_parser(clap::value_parser!(u64)),
                    )
                    .arg(
                        Arg::new("sigma")
                            .short('s')
                            .long("sigma")
                            .default_value("800.0")
                            .help("Sigma value to use for win probability")
                            .value_parser(clap::value_parser!(f32)),
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
                                    .value_parser(clap::value_parser!(PathBuf)),
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
                                    .value_parser(clap::value_parser!(PathBuf)),
                            ),
                    ),
            )
    }

    pub fn parse() -> Option<Self> {
        let matches = Self::cmd().get_matches();

        if let Some(sim) = matches.subcommand_matches("simulate") {
            return Some(Self::Simulate {
                file: sim.get_one::<PathBuf>("file")?.clone(),
                iterations: *sim.get_one::<u64>("iterations")?,
                sigma: *sim.get_one::<f32>("sigma")?,
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
