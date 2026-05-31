use std::{io::stdout, path::PathBuf};

use ratatui::backend::CrosstermBackend;

mod app;
mod binds;
mod entities;
mod framework;
mod tasks;

use app::App;
use framework::Root as _;
use pickems::datatypes::{Iterations, Map, Name, Set, Sigma, Teams};

pub type Msg = framework::Msg<Update, Notify, Task>;
pub type Context = framework::Context<Update, Notify, Task, State>;

#[derive(Debug, Clone, Default)]
pub struct State {
    pub path: PathBuf,
    pub opened: Option<PathBuf>,
    pub teams: Option<Map>,
    pub picks: [Option<Name>; 10],
    pub sigma: Sigma,
    pub iterations: Iterations,
    pub report_type: ReportType,
    pub report_focus: Id,
    pub modal_open: bool,
}

impl State {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
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
            Self::Teams => Self::Parameters,
            Self::Parameters => Self::Picks,
            Self::Picks => Self::Report,
            Self::Report => Self::Teams,
        }
    }

    pub const fn prev(self) -> Self {
        match self {
            Self::Teams => Self::Report,
            Self::Parameters => Self::Teams,
            Self::Picks => Self::Parameters,
            Self::Report => Self::Picks,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Screen {
    Open,
    Report,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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

impl AsRef<str> for ReportType {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PicksMode {
    #[default]
    Auto,
    Manual,
}

impl PicksMode {
    pub const fn as_str(&self) -> &str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
        }
    }
}

impl AsRef<str> for PicksMode {
    fn as_ref(&self) -> &str {
        self.as_str()
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
    PicksMode(PicksMode),
    AutoPickAssess(String),
    ManualPickAssess(String),
    SetPick { index: usize, name: Name },
    CloseModal,
    OpenReportModal,
    OpenIterationsModal,
    OpenSigmaModal,
    OpenRatingModal(Name),
    OpenNameModal(Name),
    OpenSeedModal(Name),
    OpenPicksModeModal,
    OpenPickSelectModal(usize),
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
        sigma: Sigma,
        iterations: Iterations,
        report: ReportType,
    },
    AutoPicks {
        teams: Box<Teams>,
        sigma: Sigma,
        iterations: Iterations,
    },
    ManualPicks {
        teams: Box<Teams>,
        sigma: Sigma,
        iterations: Iterations,
        three_zero: Set,
        advanced: Set,
        zero_three: Set,
    },
}

#[inline]
pub fn run(path: PathBuf) -> anyhow::Result<()> {
    App::new().run(|| State::new(path), CrosstermBackend::new(stdout()))
}
