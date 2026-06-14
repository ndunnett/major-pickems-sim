use std::{
    ffi::OsString,
    fs::{File, OpenOptions},
    io::Write as _,
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

    /// Closes and renames the temporary file to `destination`.
    fn persist(mut self, destination: &Path) -> anyhow::Result<()> {
        drop(self.file.take());
        std::fs::rename(&self.path, destination)?;
        Ok(())
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

/// Checks whether a local file exists and matches the expected Git blob hash.
fn check_local_file(path: &Path, expected_sha: &Sha1Hash) -> anyhow::Result<bool> {
    match Sha1Hash::digest_file(path) {
        Ok(actual_sha) => Ok(expected_sha == &actual_sha),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Builds a lazy iterator that downloads missing or changed data files.
pub fn data_updater(path: &Path) -> anyhow::Result<impl Iterator<Item = anyhow::Result<String>>> {
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
            let file_path = path.join(&remote.name);

            match check_local_file(&file_path, &remote.sha) {
                Ok(true) => None,
                Ok(false) => Some(
                    download_file(&file_path, &remote.download_url, &remote.sha)
                        .with_context(|| format!("failed to download '{}'", remote.name))
                        .map(|()| remote.name),
                ),
                Err(error) => Some(Err(error)),
            }
        }))
}

/// Downloads, verifies, and atomically replaces one local data file.
fn download_file(path: &Path, url: &str, expected_sha: &Sha1Hash) -> anyhow::Result<()> {
    let response = get(url)?;

    // Validate the complete response before creating or replacing a local file.
    anyhow::ensure!(
        &Sha1Hash::digest_bytes(response.as_bytes()) == expected_sha,
        "downloaded content did not match the expected SHA"
    );

    // Write to a temporary file first to avoid losing local data during errors.
    let mut file = TempFile::new(path)?;
    file.write_and_sync(response.as_bytes())?;
    file.persist(path)?;
    Ok(())
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
