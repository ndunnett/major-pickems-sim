use std::{
    ffi::OsString,
    fs::{File, OpenOptions},
    io::{ErrorKind, Write as _},
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use serde::Deserialize;
use sha1::{Digest, Sha1};

const DATA_URL: &str = "https://api.github.com/repos/ndunnett/major-pickems-sim/contents/data";
const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

/// A same-directory temporary file that cleans itself up.
struct TempFile {
    file: Option<File>,
    path: PathBuf,
}

impl TempFile {
    /// Creates a uniquely named temporary file next to `destination`.
    fn new(destination: &Path) -> anyhow::Result<Self> {
        let Some(parent) = destination.parent() else {
            anyhow::bail!("failed to get parent directory");
        };

        let Some(file_name) = destination.file_name() else {
            anyhow::bail!("failed to get file name");
        };

        let mut temp_name = OsString::from(".");
        temp_name.push(file_name);
        temp_name.push(format!(".{:x}.tmp", rand::random::<u64>()));
        let path = parent.join(temp_name);

        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;

        Ok(Self {
            file: Some(file),
            path,
        })
    }

    /// Writes all bytes and synchronizes them to the underlying storage.
    fn write_and_sync(&mut self, bytes: &[u8]) -> anyhow::Result<()> {
        let Some(file) = self.file.as_mut() else {
            anyhow::bail!("failed to write to temporary file");
        };

        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }

    /// Publishes the complete file only if `destination` is still absent.
    fn persist_noclobber(mut self, destination: &Path) -> std::io::Result<()> {
        drop(self.file.take());
        // Drop removes the temporary name after the link is created.
        std::fs::hard_link(&self.path, destination)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        drop(self.file.take());
        _ = std::fs::remove_file(&self.path);
    }
}

/// Converts a validated ASCII hexadecimal digit to its numeric value.
///
/// Callers must check that `byte` is an ASCII hexadecimal digit before calling
/// this function.
const fn hex_to_u8(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("byte is not an ASCII hexadecimal digit"),
    }
}

/// Binary representation of a SHA-1 hash returned by the GitHub Contents API.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "&str")]
struct Sha1Hash([u8; 20]);

impl Sha1Hash {
    /// Calculates the Git blob hash of the file at `path`.
    fn digest_file(path: &Path) -> std::io::Result<Self> {
        std::fs::read(path).map(Self::digest_bytes)
    }

    /// Calculates the Git blob hash of an in-memory byte sequence.
    fn digest_bytes(bytes: impl AsRef<[u8]>) -> Self {
        let bytes = bytes.as_ref();
        let mut hasher = Sha1::new();

        // The Contents API exposes Git object IDs, which hash a type-and-length
        // header before the file contents rather than hashing the bytes alone.
        hasher.update(format!("blob {}\0", bytes.len()));
        hasher.update(bytes);
        hasher.into()
    }
}

impl From<Sha1> for Sha1Hash {
    /// Finalizes a SHA-1 hasher and stores its digest as a fixed-size byte array.
    fn from(hasher: Sha1) -> Self {
        Self(hasher.finalize().into())
    }
}

impl TryFrom<&str> for Sha1Hash {
    type Error = anyhow::Error;

    /// Parses the 40-character hexadecimal SHA-1 format returned by GitHub.
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        anyhow::ensure!(
            s.len() == 40,
            "SHA1 hash string is not the correct length: '{s}'"
        );

        anyhow::ensure!(
            s.bytes().all(|b| b.is_ascii_hexdigit()),
            "SHA1 hash string contains non-hexdigit bytes: '{s}'"
        );

        // Each pair of hexadecimal characters represents one byte of the hash.
        let b = s.as_bytes();
        let sha = std::array::from_fn(|i| (hex_to_u8(b[i * 2]) << 4) + hex_to_u8(b[i * 2 + 1]));
        Ok(Self(sha))
    }
}

/// Metadata required to compare and download one remote data file.
#[derive(Debug, Clone, Deserialize)]
struct RemoteRecord {
    name: String,
    download_url: String,
    sha: Sha1Hash,
}

pub enum FileUpdate {
    Downloaded(String),
    KeptLocal(String),
}

/// Downloads missing data files and reports conflicts without overwriting them.
pub fn data_updater(
    path: &Path,
) -> anyhow::Result<impl Iterator<Item = anyhow::Result<FileUpdate>>> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }

    if !path.is_dir() {
        anyhow::bail!("path exists and is not a directory: {}", path.display());
    }

    Ok(get(DATA_URL)?
        .json::<Vec<RemoteRecord>>()?
        .into_iter()
        .filter_map(move |remote| {
            update_file(path, &remote.name, &remote.sha, || {
                Ok(get(&remote.download_url)?.into_bytes())
            })
            .with_context(|| format!("failed to update '{}'", remote.name))
            .transpose()
        }))
}

fn update_file(
    directory: &Path,
    name: &str,
    expected_sha: &Sha1Hash,
    download: impl FnOnce() -> anyhow::Result<Vec<u8>>,
) -> anyhow::Result<Option<FileUpdate>> {
    let path = directory.join(name);
    match path.symlink_metadata() {
        Ok(meta) => {
            if meta.is_file() && Sha1Hash::digest_file(&path)? == *expected_sha {
                return Ok(None);
            }
            return Ok(Some(FileUpdate::KeptLocal(name.to_owned())));
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }

    let bytes = download()?;
    anyhow::ensure!(
        &Sha1Hash::digest_bytes(&bytes) == expected_sha,
        "downloaded content did not match the expected SHA"
    );

    let mut file = TempFile::new(&path)?;
    file.write_and_sync(&bytes)?;
    // A user may have saved to this name while the download was in flight.
    match file.persist_noclobber(&path) {
        Ok(()) => Ok(Some(FileUpdate::Downloaded(name.to_owned()))),
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
            Ok(Some(FileUpdate::KeptLocal(name.to_owned())))
        }
        Err(e) => Err(e.into()),
    }
}

/// Sends a GET request with the application's user agent and validates its status.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> std::io::Result<Self> {
            let path = std::env::temp_dir().join(format!("pickems-{:x}", rand::random::<u64>()));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn update_keeps_edits_and_installs_missing_files() -> anyhow::Result<()> {
        let dir = TestDir::new()?;
        let official = b"official ratings";
        let sha = Sha1Hash::digest_bytes(official);
        fs::write(dir.0.join("edited.toml"), b"local ratings")?;

        let kept = update_file(&dir.0, "edited.toml", &sha, || {
            panic!("unexpected download")
        })?;
        assert!(matches!(kept, Some(FileUpdate::KeptLocal(name)) if name == "edited.toml"));
        assert_eq!(fs::read(dir.0.join("edited.toml"))?, b"local ratings");

        let added = update_file(&dir.0, "missing.toml", &sha, || Ok(official.to_vec()))?;
        assert!(matches!(added, Some(FileUpdate::Downloaded(name)) if name == "missing.toml"));
        assert_eq!(fs::read(dir.0.join("missing.toml"))?, official);
        assert!(
            update_file(&dir.0, "missing.toml", &sha, || panic!(
                "unexpected download"
            ))?
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn save_during_download_is_preserved() -> anyhow::Result<()> {
        let dir = TestDir::new()?;
        let path = dir.0.join("ratings.toml");
        let sha = Sha1Hash::digest_bytes(b"official");
        let result = update_file(&dir.0, "ratings.toml", &sha, || {
            fs::write(&path, b"user edits")?;
            Ok(b"official".to_vec())
        })?;
        assert!(matches!(result, Some(FileUpdate::KeptLocal(_))));
        assert_eq!(fs::read(path)?, b"user edits");
        Ok(())
    }
}
