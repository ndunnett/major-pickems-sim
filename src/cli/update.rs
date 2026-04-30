use std::{ffi::OsString, fmt::Write as _, io::Write as _, path::PathBuf};

use serde::Deserialize;
use sha1::{Digest, Sha1};

const DATA_URL: &str = "https://api.github.com/repos/ndunnett/major-pickems-sim/contents/data";
const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone)]
struct LocalRecord {
    name: OsString,
    path: PathBuf,
    sha: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RemoteRecord {
    name: String,
    download_url: String,
    sha: String,
}

/// Update local TOML data files from the repository data directory.
pub fn run(path: &PathBuf) -> anyhow::Result<()> {
    let local_records = get_local_records(path)?;
    let remote_records = get_remote_records()?;

    for remote in remote_records {
        if let Some(local) = local_records
            .iter()
            .find(|local| local.name.eq_ignore_ascii_case(&remote.name))
        {
            if remote.sha != local.sha {
                println!("Updating {}...", remote.name);
                download_file(&local.path, &remote.download_url)?;
            }

            continue;
        }

        println!("Downloading {}...", remote.name);
        let new_path = path.join(remote.name);
        download_file(&new_path, &remote.download_url)?;
    }

    Ok(())
}

fn get_local_records(path: &PathBuf) -> anyhow::Result<Vec<LocalRecord>> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }

    if !path.is_dir() {
        anyhow::bail!("path exists and is not a directory: {}", path.display());
    }

    let mut records = Vec::new();

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;

        if entry
            .path()
            .extension()
            .is_none_or(|ext| !ext.eq_ignore_ascii_case("toml"))
        {
            continue;
        }

        // Git blob object IDs hash a header plus contents, not just the raw
        // file bytes. GitHub's contents API exposes this blob SHA.
        let file = std::fs::read(entry.path())?;
        let mut buffer = format!("blob {}\0", file.len()).into_bytes();
        buffer.extend(file);

        let digest = Sha1::digest(buffer);
        let mut sha = String::with_capacity(40);

        for byte in digest {
            write!(sha, "{byte:02x}")?;
        }

        records.push(LocalRecord {
            name: entry.file_name(),
            path: entry.path(),
            sha,
        });
    }

    Ok(records)
}

fn get_remote_records() -> anyhow::Result<Vec<RemoteRecord>> {
    Ok(get(DATA_URL)?.json()?)
}

fn download_file(path: &PathBuf, url: &str) -> anyhow::Result<()> {
    let response = get(url)?;

    std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?
        .write_all(response.as_bytes())?;

    Ok(())
}

fn get(url: &str) -> anyhow::Result<minreq::Response> {
    let response = minreq::get(url)
        .with_header("User-Agent", USER_AGENT)
        .send()?;

    if !(200..300).contains(&response.status_code) {
        anyhow::bail!(
            "GET {url} failed with status {} {}",
            response.status_code,
            response.reason_phrase
        );
    }

    Ok(response)
}
