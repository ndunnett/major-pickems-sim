use std::{ffi::OsString, fmt::Write as _, io::Write as _, path::Path};

use serde::Deserialize;
use sha1::{Digest, Sha1};

const DATA_URL: &str = "https://api.github.com/repos/ndunnett/major-pickems-sim/contents/data";
const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone)]
struct LocalRecord {
    name: OsString,
    sha: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RemoteRecord {
    name: String,
    download_url: String,
    sha: String,
}

pub fn data_updater(path: &Path) -> anyhow::Result<impl Iterator<Item = anyhow::Result<String>>> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }

    if !path.is_dir() {
        anyhow::bail!("path exists and is not a directory: {}", path.display());
    }

    let mut local_records = Vec::new();

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;

        if entry
            .path()
            .extension()
            .is_none_or(|ext| !ext.eq_ignore_ascii_case("toml"))
        {
            continue;
        }

        let file = std::fs::read(entry.path())?;
        let mut buffer = format!("blob {}\0", file.len()).into_bytes();
        buffer.extend(file);

        let digest = Sha1::digest(buffer);
        let mut sha = String::with_capacity(40);

        for byte in digest {
            write!(sha, "{byte:02x}")?;
        }

        local_records.push(LocalRecord {
            name: entry.file_name(),
            sha,
        });
    }

    let remote_records: Vec<RemoteRecord> = get(DATA_URL)?.json()?;

    Ok(remote_records.into_iter().filter_map(move |remote| {
        if local_records
            .iter()
            .any(|local| local.name.eq_ignore_ascii_case(&remote.name) && remote.sha == local.sha)
        {
            None
        } else {
            Some(download_file(path.join(&remote.name), &remote.download_url).map(|()| remote.name))
        }
    }))
}

fn download_file<P: AsRef<Path>>(path: P, url: &str) -> anyhow::Result<()> {
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
        .with_timeout(5)
        .send()?;

    if !(200..300).contains(&response.status_code) {
        anyhow::bail!(
            "Request to {url} failed: {} {}",
            response.status_code,
            response.reason_phrase
        );
    }

    Ok(response)
}
