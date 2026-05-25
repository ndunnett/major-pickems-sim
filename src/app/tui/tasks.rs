use std::path::{Path, PathBuf};

use pickems::{
    datatypes::Teams,
    reporting::{BasicReport, PicksReport, Report as _, ReportAll, StrengthReport},
    simulation::Simulation,
};

use crate::app::{tui::ReportType, update::data_updater};

pub fn update_data(path: &Path) -> anyhow::Result<impl Iterator<Item = String>> {
    Ok(data_updater(path)?.filter_map(Result::ok))
}

pub fn load_file_list(path: &Path) -> anyhow::Result<impl Iterator<Item = PathBuf>> {
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    if !path.exists() {
        anyhow::bail!("path doesn't exist: {}", path.to_string_lossy());
    }

    if !path.is_dir() {
        anyhow::bail!(
            "path exists and is not a directory: {}",
            path.to_string_lossy()
        );
    }

    Ok(std::fs::read_dir(path)?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
                .then(|| entry.path())
        }))
}

pub fn run_simulation(teams: Teams, sigma: f32, iterations: u64, report: ReportType) -> String {
    let sim = Simulation::new(teams, sigma, iterations);

    match report {
        ReportType::Basic => sim.run(BasicReport::default()).format(&sim),
        ReportType::Strength => sim.run(StrengthReport::default()).format(&sim),
        ReportType::Picks => sim.run(PicksReport::default()).format(&sim),
        ReportType::All => sim.run(ReportAll::default()).format(&sim),
    }
}
