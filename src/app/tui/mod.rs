use std::path::PathBuf;

#[inline]
pub fn run(path: PathBuf) -> anyhow::Result<()> {
    println!("{}", path.into_os_string().display());
    todo!()
}
