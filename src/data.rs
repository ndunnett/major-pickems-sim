use std::{collections::BTreeMap, fmt::Debug, fs::read_to_string, io::Write, path::PathBuf};

use anyhow::anyhow;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::wizard::Wizard;

pub type TeamSeed = u16;

/// Struct to store team data, used solely for serde.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamData {
    pub seed: TeamSeed,
    pub rating: i16,
}

/// Struct to store collection of teams, used solely for serde.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamDataMap(BTreeMap<String, TeamData>);

impl<I: IntoIterator<Item = (String, TeamData)>> From<I> for TeamDataMap {
    fn from(teams: I) -> Self {
        Self(teams.into_iter().collect())
    }
}

impl TryFrom<TeamDataMap> for ([String; 16], [i16; 16]) {
    type Error = anyhow::Error;

    fn try_from(teams_map: TeamDataMap) -> Result<Self, Self::Error> {
        let teams = teams_map
            .0
            .into_iter()
            .sorted_by(|(_, a), (_, b)| a.seed.cmp(&b.seed))
            .map(|(name, data)| (name, data.rating))
            .collect::<Vec<_>>();

        if teams.len() == 16 {
            let ratings = teams
                .iter()
                .map(|(_, rating)| *rating)
                .collect_array()
                .ok_or_else(|| anyhow!("failed to allocate array"))?;

            let names = teams
                .into_iter()
                .map(|(name, _)| name)
                .collect_array()
                .ok_or_else(|| anyhow!("failed to allocate array"))?;

            Ok((names, ratings))
        } else {
            Err(anyhow!(
                "there must be exactly 16 teams (only {} teams recognised)",
                teams.len()
            ))
        }
    }
}

/// Parses a TOML file into a SoA of team information.
pub fn parse_toml(filepath: PathBuf) -> anyhow::Result<([String; 16], [i16; 16])> {
    <([String; 16], [i16; 16])>::try_from(toml::from_str::<TeamDataMap>(&read_to_string(
        filepath,
    )?)?)
}

/// Writes a map of teams into a TOML file.
pub fn write_toml(filepath: PathBuf, teams: &TeamDataMap) -> anyhow::Result<()> {
    let contents = toml::to_string_pretty(teams)?;

    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(filepath)?
        .write_all(contents.as_bytes())?;

    Ok(())
}

/// Format a SoA of team information into a table suitable to print.
pub fn format_teams(teams: &([String; 16], [i16; 16])) -> String {
    let mut out = Vec::new();
    out.push(format!("{:<4}  {:<18}{:>6}", "Seed", "Team", "Rating"));

    for (seed, (name, rating)) in teams.0.iter().zip(teams.1.iter()).enumerate() {
        out.push(format!(
            "{:<4}  {:<18}{:>6}",
            format!("{}.", seed + 1),
            name,
            rating
        ));
    }

    out.join("\n")
}

/// Print input data loaded from a TOML file.
pub fn inspect(filepath: PathBuf) -> anyhow::Result<()> {
    println!("{}", format_teams(&parse_toml(filepath)?));
    Ok(())
}

/// Run wizard to generate team data and write to a TOML file.
pub fn wizard(filepath: PathBuf) -> anyhow::Result<()> {
    if let Some(teams) = Wizard::run()? {
        write_toml(filepath.clone(), &teams)?;

        println!(
            "Generated input data, saved to '{}':\n",
            filepath.canonicalize()?.display(),
        );

        inspect(filepath)?;
    } else {
        println!("Exited wizard without saving.",);
    }

    Ok(())
}
