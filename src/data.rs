use std::{
    fmt::{Debug, Display, Formatter},
    fs::read_to_string,
    hash::{Hash, Hasher},
    path::PathBuf,
};

use anyhow::anyhow;
use itertools::Itertools;
use serde_json::Value;

/// Struct to represent each team, including rating and seeding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Team {
    pub id: u8,
    pub name: String,
    pub seed: u8,
    pub rating: i32,
}

impl Hash for Team {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Display for Team {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

/// Parses a .json file into a vec of teams.
pub fn parse_json(filepath: PathBuf) -> anyhow::Result<Vec<Team>> {
    let mut teams = Vec::new();
    let mut id_iter = 0..u8::MAX;

    let contents = read_to_string(filepath)?;
    let json: Value = serde_json::from_str(&contents)?;
    let object = json.as_object().ok_or(anyhow!("failed to parse .json"))?;
    let list = object.iter().sorted_by(|(_, a), (_, b)| {
        a["seed"]
            .as_number()
            .unwrap()
            .as_u64()
            .unwrap()
            .cmp(&b["seed"].as_number().unwrap().as_u64().unwrap())
    });

    for (key, value) in list {
        let id = id_iter.next().ok_or(anyhow!("exhausted id values"))?;
        let name = key.to_string();

        let seed = value["seed"]
            .as_number()
            .ok_or(anyhow!("failed to parse seed as number"))?
            .as_u64()
            .ok_or(anyhow!("failed to cast seed to u64"))? as u8;

        let rating = value["rating"]
            .as_number()
            .ok_or(anyhow!("failed to parse rating as number"))?
            .as_i64()
            .ok_or(anyhow!("failed to cast rating to i64"))? as i32;

        teams.push(Team {
            id,
            name,
            seed,
            rating,
        });
    }

    Ok(teams)
}
