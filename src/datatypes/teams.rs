use std::collections::BTreeMap;
use std::{fs::read_to_string, io::Write, path::PathBuf};

use anyhow::anyhow;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::datatypes::{Index, Name, Rating, Seed, Set};

/// Struct to store team data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub seed: Seed,
    pub rating: Rating,
}

/// Struct to store a collection of teams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Map(BTreeMap<Name, Team>);

impl Map {
    pub fn parse_toml(filepath: PathBuf) -> anyhow::Result<Self> {
        Ok(toml::from_str::<Self>(&read_to_string(filepath)?)?)
    }

    pub fn write_toml(&self, filepath: PathBuf) -> anyhow::Result<()> {
        let contents = toml::to_string_pretty(self)?;

        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(filepath)?
            .write_all(contents.as_bytes())?;

        Ok(())
    }
}

impl<I: IntoIterator<Item = (Name, Team)>> From<I> for Map {
    fn from(teams: I) -> Self {
        Self(teams.into_iter().collect())
    }
}

/// Struct to efficiently store a collection of teams as a struct of arrays.
#[derive(Debug, Clone)]
pub struct Teams {
    pub names: [Name; 16],
    pub ratings: [Rating; 16],
}

impl Teams {
    /// Produce dummy data for testing purposes.
    #[must_use]
    pub fn dummy() -> Self {
        Self {
            names: (0..16)
                .map(|i| Name::try_new(format!("Team {}", i + 1)).unwrap())
                .collect_array()
                .unwrap(),
            ratings: (0..16)
                .map(|i| Rating::try_new(2000 - 50 * i).unwrap())
                .collect_array()
                .unwrap(),
        }
    }

    pub fn parse_toml(filepath: PathBuf) -> anyhow::Result<Self> {
        Self::try_from(Map::parse_toml(filepath)?)
    }

    pub fn write_toml(&self, filepath: PathBuf) -> anyhow::Result<()> {
        let map = Map::from(self);
        map.write_toml(filepath)
    }
}

impl TryFrom<Map> for Teams {
    type Error = anyhow::Error;

    fn try_from(teams_map: Map) -> Result<Self, Self::Error> {
        if teams_map.0.len() != 16 {
            return Err(anyhow!(
                "there must be exactly 16 teams ({} teams recognised)",
                teams_map.0.len(),
            ));
        }

        let set = teams_map
            .0
            .values()
            .map(|team| Index::from(team.seed))
            .collect::<Set>();

        if set != Set::full() {
            for i in Index::iter_all() {
                if !set.contains(i) {
                    return Err(anyhow!("missing seed: {i}"));
                }
            }

            let indices = teams_map
                .0
                .values()
                .map(|team| Index::from(team.seed))
                .collect::<Vec<_>>();

            for i in Index::iter_all() {
                if indices.iter().filter(|&&index| index == i).count() > 1 {
                    return Err(anyhow!("duplicate seed: {i}"));
                }
            }
        }

        let teams = teams_map
            .0
            .into_iter()
            .sorted_by(|(_, a), (_, b)| a.seed.cmp(&b.seed))
            .map(|(name, data)| (name, data.rating))
            .collect::<Vec<_>>();

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

        Ok(Self { names, ratings })
    }
}

impl From<&Teams> for Map {
    fn from(teams: &Teams) -> Self {
        Self(
            (0..16)
                .map(|i| {
                    (
                        teams.names[i].clone(),
                        Team {
                            seed: Seed::try_new(i as u16 + 1).unwrap(),
                            rating: teams.ratings[i],
                        },
                    )
                })
                .collect(),
        )
    }
}
