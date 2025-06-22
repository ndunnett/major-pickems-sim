use std::{
    collections::BTreeMap,
    fmt::{Debug, Display, Formatter},
    fs::read_to_string,
    hash::{Hash, Hasher},
    path::PathBuf,
};

use anyhow::anyhow;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

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
