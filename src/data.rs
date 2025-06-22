use std::{
    collections::BTreeMap,
    fmt::{Debug, Display, Formatter},
    fs::read_to_string,
    hash::{Hash, Hasher},
    io::Write,
    path::PathBuf,
};

use anyhow::anyhow;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::wizard::Wizard;

/// Struct to represent each team, including rating and seeding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Team {
    pub name: String,
    pub seed: u8,
    pub rating: i16,
}

impl Team {
    /// Returns what should be the index of this team within an array sorted by seed.
    pub const fn index(&self) -> usize {
        self.seed as usize - 1
    }
}

impl Hash for Team {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.seed.hash(state);
    }
}

impl Display for Team {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

/// Struct to store team data, used solely for serde.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamData {
    pub seed: u8,
    pub rating: i16,
}

impl From<&Team> for TeamData {
    fn from(team: &Team) -> Self {
        Self {
            seed: team.seed,
            rating: team.rating,
        }
    }
}

/// Struct to store collection of teams, used solely for serde.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamDataMap(BTreeMap<String, TeamData>);

impl From<&[Team]> for TeamDataMap {
    fn from(teams: &[Team]) -> Self {
        Self(
            teams
                .iter()
                .map(|team| (team.name.clone(), TeamData::from(team)))
                .collect(),
        )
    }
}

impl TryFrom<TeamDataMap> for [Team; 16] {
    type Error = anyhow::Error;

    fn try_from(teams_map: TeamDataMap) -> Result<Self, Self::Error> {
        let teams = teams_map
            .0
            .into_iter()
            .map(|(name, data)| Team {
                name,
                seed: data.seed,
                rating: data.rating,
            })
            .sorted_by(|a, b| a.seed.cmp(&b.seed))
            .collect::<Vec<_>>();

        if teams.len() == 16 {
            Ok(teams
                .try_into()
                .map_err(|_| anyhow!("failed to allocate array"))?)
        } else {
            Err(anyhow!("need to have exactly 16 teams"))
        }
    }
}

/// Parses a TOML file into an array of teams.
pub fn parse_toml(filepath: PathBuf) -> anyhow::Result<[Team; 16]> {
    <[Team; 16]>::try_from(toml::from_str::<TeamDataMap>(&read_to_string(filepath)?)?)
}

/// Writes an iterator of teams into a TOML file.
pub fn write_toml(filepath: PathBuf, teams: &[Team]) -> anyhow::Result<()> {
    let teams_map = TeamDataMap::from(teams);
    let contents = toml::to_string_pretty(&teams_map)?;

    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(filepath)?
        .write_all(contents.as_bytes())?;

    Ok(())
}

/// Format an iterator of teams into a table suitable to print.
pub fn format_teams(teams: &[Team]) -> String {
    let mut out = Vec::new();
    out.push(format!("{:<4}  {:<18}{:>6}", "Seed", "Team", "Rating"));

    for team in teams {
        out.push(format!(
            "{:<4}  {:<18}{:>6}",
            format!("{}.", team.seed),
            team.name,
            team.rating
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
            "Generated input data, saved to '{}':\n\n{}",
            filepath.canonicalize()?.display(),
            format_teams(&teams)
        );
    } else {
        println!("Exited wizard without saving.",);
    }

    Ok(())
}
