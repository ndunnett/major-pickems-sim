use std::{io::stdout, path::PathBuf};

use ratatui::backend::CrosstermBackend;

mod app;
mod binds;
mod entities;
mod framework;
mod tasks;

use app::App;
use framework::Root as _;
use pickems::datatypes::{Map, Teams};

pub type Msg = framework::Msg<Update, Notify, Task>;
pub type Context = framework::Context<Update, Notify, Task, State>;

#[derive(Debug, Clone, Default)]
pub struct State {
    pub path: PathBuf,
    pub opened: Option<PathBuf>,
    pub teams: Option<Map>,
    pub sigma: f32,
    pub iterations: u64,
    pub report_type: ReportType,
    pub report_focus: Id,
}

impl State {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            sigma: 800.0,
            iterations: 1_000_000,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum Id {
    #[default]
    Report,
    Teams,
    Picks,
    Parameters,
}

impl Id {
    pub const fn next(self) -> Self {
        match self {
            Self::Teams => Self::Picks,
            Self::Picks => Self::Report,
            Self::Report => Self::Parameters,
            Self::Parameters => Self::Teams,
        }
    }

    pub const fn prev(self) -> Self {
        match self {
            Self::Teams => Self::Parameters,
            Self::Picks => Self::Teams,
            Self::Report => Self::Picks,
            Self::Parameters => Self::Report,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Screen {
    Open,
    Report,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub enum ReportType {
    #[default]
    Basic,
    Strength,
    Picks,
    All,
}

impl ReportType {
    pub const fn as_str(&self) -> &str {
        match self {
            Self::Basic => "basic",
            Self::Strength => "strength",
            Self::Picks => "picks",
            Self::All => "all",
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Update {
    ChangeScreen(Screen),
    ChangePath(PathBuf),
    LoadFileList(PathBuf),
    LoadDataFile(PathBuf),
    ReportContent(String),
    DataOrParams,
    AutoPicks(String),
}

#[derive(Debug, Clone, Copy)]
pub enum Notify {
    Select,
    Todo,
}

#[derive(Debug, Clone)]
pub enum Task {
    UpdateData {
        path: PathBuf,
    },
    RunSimulation {
        teams: Box<Teams>,
        sigma: f32,
        iterations: u64,
        report: ReportType,
    },
    AutoPicks {
        teams: Box<Teams>,
        sigma: f32,
        iterations: u64,
    },
}

#[inline]
pub fn run(path: PathBuf) -> anyhow::Result<()> {
    App::new().run(|| State::new(path), CrosstermBackend::new(stdout()))
}
