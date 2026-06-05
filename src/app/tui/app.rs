use std::path::PathBuf;

use ratatui::{
    Frame,
    crossterm::event::{Event, KeyCode, KeyModifiers},
    layout::Rect,
};

use pickems::{
    datatypes::{Index, Iterations, Map, Name, Set, Sigma, Teams},
    reporting::{AssessReport, BasicReport, PicksReport, Report as _, ReportAll, StrengthReport},
    simulation::Simulation,
};

use crate::app::{
    tui::{
        ReportType, Screen, State, Task, Update, binds,
        entities::{InputModal, OpenScreen, ReportScreen},
        framework::{Entity, Root},
    },
    update::data_updater,
};

type Context = crate::app::tui::framework::Context<Update, Task, State>;
type Msg = crate::app::tui::framework::Msg<Update, Task>;

pub struct App {
    open: OpenScreen,
    report: ReportScreen,
    active: Screen,
    input_modal: Option<InputModal>,
}

impl App {
    pub fn new() -> Self {
        Self {
            open: OpenScreen::new(),
            report: ReportScreen::new(),
            active: Screen::Open,
            input_modal: None,
        }
    }

    fn update_data_files(path: PathBuf) -> Update {
        if data_updater(&path).is_ok_and(|it| it.filter_map(Result::ok).count() > 0) {
            Update::LoadFileList(path)
        } else {
            Update::Todo
        }
    }

    fn run_simulation(
        teams: Box<Teams>,
        sigma: Sigma,
        iterations: Iterations,
        report: ReportType,
    ) -> String {
        let sim = Simulation::new(*teams, sigma, iterations);

        match report {
            ReportType::Basic => sim.run(BasicReport::default()).format(&sim),
            ReportType::Strength => sim.run(StrengthReport::default()).format(&sim),
            ReportType::Picks => sim.run(PicksReport::default()).format(&sim),
            ReportType::All => sim.run(ReportAll::default()).format(&sim),
        }
    }

    fn assess_picks(
        teams: Box<Teams>,
        sigma: Sigma,
        iterations: Iterations,
        three_zero: Set,
        advanced: Set,
        zero_three: Set,
    ) -> String {
        let sim = Simulation::new(*teams, sigma, iterations);
        sim.run(AssessReport::new(three_zero, advanced, zero_three))
            .format(&sim)
    }

    fn load_data(cx: &mut Context, teams: Map, path: PathBuf) {
        cx.opened = Some(path);
        cx.teams = Some(teams);
        cx.picks = Default::default();
        cx.update(Update::DataOrParams);
        cx.update(Update::ChangeScreen(Screen::Report));
    }

    fn set_pick(cx: &mut Context, index: usize, name: Name) {
        for pick in &mut cx.picks {
            _ = pick.take_if(|pick| pick == &name);
        }

        cx.picks[index] = Some(name);

        if cx.picks.iter().any(Option::is_none) {
            cx.update(Update::ManualPickAssess(String::new()));
            return;
        }

        let Some(teams) = &cx.teams else { return };
        let Ok(teams_soa) = Teams::try_from(teams.clone()) else {
            return;
        };

        let collect_set = |slice: &[Option<Name>]| {
            slice
                .iter()
                .filter_map(|name| {
                    let Some(name) = name else { return None };
                    Some(Index::from(teams[name].seed))
                })
                .collect::<Set>()
        };

        cx.task(Task::ManualPicks {
            teams: Box::new(teams_soa),
            sigma: cx.sigma,
            iterations: cx.iterations,
            three_zero: collect_set(&cx.picks[..2]),
            advanced: collect_set(&cx.picks[2..8]),
            zero_three: collect_set(&cx.picks[8..]),
        });
    }

    fn spawn_simulation_task(cx: &mut Context) {
        if let Some(teams) = cx.teams.clone()
            && let Ok(teams_soa) = Teams::try_from(teams)
        {
            cx.task(Task::RunSimulation {
                teams: Box::new(teams_soa),
                sigma: cx.sigma,
                iterations: cx.iterations,
                report: cx.report_type,
            });
        } else {
            cx.update(Update::Todo);
        }
    }

    fn open_modal(&mut self, cx: &mut Context, modal: InputModal) {
        self.input_modal = Some(modal);
        cx.modal_open = true;
    }
}

impl Root<Update, Task, State> for App {
    const MAX_FPS: u64 = 120;

    fn handle_task(task: Task) -> Option<Msg> {
        match task {
            Task::UpdateData { path } => Some(Msg::Update(Self::update_data_files(path))),
            Task::RunSimulation {
                teams,
                sigma,
                iterations,
                report,
            } => Some(Msg::Update(Update::ReportContent(Self::run_simulation(
                teams, sigma, iterations, report,
            )))),
            Task::AutoPicks {
                teams,
                sigma,
                iterations,
            } => Some(Msg::Update(Update::AutoPickAssess(Self::run_simulation(
                teams,
                sigma,
                iterations,
                ReportType::Picks,
            )))),
            Task::ManualPicks {
                teams,
                sigma,
                iterations,
                three_zero,
                advanced,
                zero_three,
            } => Some(Msg::Update(Update::ManualPickAssess(Self::assess_picks(
                teams, sigma, iterations, three_zero, advanced, zero_three,
            )))),
        }
    }
}

impl Entity<Update, Task, State> for App {
    fn dispatch_event(&mut self, cx: &mut Context, event: &Event) -> Option<Msg> {
        if let Some(input_modal) = &mut self.input_modal {
            input_modal.dispatch_event(cx, event)
        } else {
            match self.active {
                Screen::Open => self.open.dispatch_event(cx, event),
                Screen::Report => self.report.dispatch_event(cx, event),
            }
            .map_or_else(|| self.handle_event(cx, event), Some)
        }
    }

    fn on_key_press(
        &mut self,
        _cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::EOF | binds::QUIT => Some(Msg::Quit),
            binds::REDRAW => Some(Msg::Redraw),
            _ => None,
        }
    }

    fn on_load(&mut self, cx: &mut Context) {
        cx.update(Update::LoadFileList(cx.path.clone()));
        cx.task(Task::UpdateData {
            path: cx.path.clone(),
        });
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        match msg {
            Update::Todo => {}
            Update::ChangeScreen(screen) => {
                self.active = screen;
            }
            Update::ChangePath(path) => {
                cx.path = path;
                cx.update(Update::LoadFileList(cx.path.clone()));
                cx.update(Update::ChangeScreen(Screen::Open));
            }
            Update::LoadDataFile(path) => {
                if let Ok(teams) = Map::parse_toml(path.clone()) {
                    Self::load_data(cx, teams, path);
                } else {
                    cx.update(Update::Todo);
                }
            }
            Update::NewInputData => {
                let teams = Map::from(&Teams::dummy());
                let path = cx.path.join("new_data_file.toml");
                Self::load_data(cx, teams, path);
            }
            Update::DataSaved(path) => {
                cx.opened = Some(path);
                cx.update(Update::LoadFileList(cx.path.clone()));
            }
            Update::SetPick { index, name } => Self::set_pick(cx, index, name),
            Update::ReportContent(..)
            | Update::AutoPickAssess(..)
            | Update::ManualPickAssess(..)
            | Update::PicksMode(..) => {
                self.report.update(cx, msg);
            }
            Update::DataOrParams => {
                self.report.update(cx, msg);
                Self::spawn_simulation_task(cx);
            }
            Update::LoadFileList(..) => self.open.update(cx, msg),
            Update::CloseModal => {
                self.input_modal = None;
                cx.modal_open = false;
            }
            Update::OpenReportModal => self.open_modal(cx, InputModal::report(cx)),
            Update::OpenIterationsModal => self.open_modal(cx, InputModal::iterations(cx)),
            Update::OpenSigmaModal => self.open_modal(cx, InputModal::sigma(cx)),
            Update::OpenRatingModal(name) => self.open_modal(cx, InputModal::rating(cx, name)),
            Update::OpenNameModal(name) => self.open_modal(cx, InputModal::name(name)),
            Update::OpenSeedModal(name) => self.open_modal(cx, InputModal::seed(cx, name)),
            Update::OpenPicksModeModal => self.open_modal(cx, InputModal::picks_mode()),
            Update::OpenPickSelectModal(n) => self.open_modal(cx, InputModal::pick_select(cx, n)),
            Update::OpenSaveModal => self.open_modal(cx, InputModal::save(cx)),
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        match self.active {
            Screen::Open => self.open.render(cx, frame, area),
            Screen::Report => self.report.render(cx, frame, area),
        }

        if let Some(input_modal) = &mut self.input_modal {
            input_modal.render(cx, frame, area);
        }
    }
}
