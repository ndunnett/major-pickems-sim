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
        entities::{InputModal, OpenScreen, ReportScreen, Toast, ToastKind, ToastMessage},
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
    toast: Toast,
}

impl App {
    pub fn new() -> Self {
        Self {
            open: OpenScreen::new(),
            report: ReportScreen::new(),
            active: Screen::Open,
            input_modal: None,
            toast: Toast::new(),
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
            Task::UpdateData { path } => Some(Msg::Update(update_data_files(path))),
            Task::RunSimulation {
                request_id,
                teams,
                sigma,
                iterations,
                report,
            } => Some(Msg::Update(Update::ReportContent(
                request_id,
                run_simulation(teams, sigma, iterations, report),
            ))),
            Task::AutoPicks {
                request_id,
                teams,
                sigma,
                iterations,
            } => Some(Msg::Update(Update::AutoPickAssess(
                request_id,
                run_simulation(teams, sigma, iterations, ReportType::Picks),
            ))),
            Task::ManualPicks {
                request_id,
                teams,
                sigma,
                iterations,
                three_zero,
                advanced,
                zero_three,
            } => Some(Msg::Update(Update::ManualPickAssess(
                request_id,
                assess_picks(teams, sigma, iterations, three_zero, advanced, zero_three),
            ))),
        }
    }
}

impl Entity<Update, Task, State> for App {
    fn dispatch_event(&mut self, cx: &mut Context, event: &Event) -> Option<Msg> {
        self.toast.dispatch_event(cx, event).map_or_else(
            || {
                if let Some(input_modal) = &mut self.input_modal {
                    input_modal.dispatch_event(cx, event)
                } else {
                    match self.active {
                        Screen::Open => self.open.dispatch_event(cx, event),
                        Screen::Report => self.report.dispatch_event(cx, event),
                    }
                    .map_or_else(|| self.handle_event(cx, event), Some)
                }
            },
            Some,
        )
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

    fn on_resize(&mut self, cx: &mut Context, width: u16, height: u16) -> Option<Msg> {
        self.report.on_resize(cx, width, height)
    }

    fn on_tick(&mut self, cx: &mut Context) {
        self.toast.on_tick(cx);

        match self.active {
            Screen::Open => self.open.on_tick(cx),
            Screen::Report => self.report.on_tick(cx),
        }
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        match msg {
            Update::ChangeScreen(screen) => {
                self.active = screen;
            }
            Update::ChangePath(path) => {
                cx.path = path;
                cx.update(Update::LoadFileList(cx.path.clone()));
                cx.update(Update::ChangeScreen(Screen::Open));
            }
            Update::LoadDataFile(path) => match Map::parse_toml(path.clone()) {
                Ok(teams) => load_data(cx, teams, path),
                Err(e) => cx.update(Update::ErrorToast(format!(
                    "Failed to load {}: {e}",
                    path.display()
                ))),
            },
            Update::NewInputData => {
                let teams = Map::from(&Teams::dummy());
                let path = cx.path.join("new_data_file.toml");
                load_data(cx, teams, path);
            }
            Update::DataSaved(path) => {
                cx.opened = Some(path);
                cx.update(Update::LoadFileList(cx.path.clone()));
            }
            Update::DataFilesUpdated { path, files } => {
                if !files.is_empty() {
                    cx.update(Update::LoadFileList(path));
                    cx.update(Update::InfoToast(format!(
                        "Downloaded data files:\n- {}",
                        files.join("\n- ")
                    )));
                }
            }
            Update::SetPick { index, name } => set_pick(cx, index, name),
            Update::ErrorToast(text) => self.toast.push(
                ToastMessage {
                    kind: ToastKind::Error,
                    text,
                },
                cx.tick(),
            ),
            Update::InfoToast(text) => self.toast.push(
                ToastMessage {
                    kind: ToastKind::Info,
                    text,
                },
                cx.tick(),
            ),
            Update::ReportContent(id, _) if id != cx.report_request_id => {}
            Update::AutoPickAssess(id, _) if id != cx.auto_picks_request_id => {}
            Update::ManualPickAssess(id, _) if id != cx.manual_picks_request_id => {}
            Update::ReportContent(..)
            | Update::AutoPickAssess(..)
            | Update::ManualPickAssess(..)
            | Update::PicksMode(..) => {
                self.report.update(cx, msg);
            }
            Update::RefreshFull => {
                self.report.update(cx, Update::RefreshFull);
                refresh_full(cx);
            }
            Update::RefreshTeamValues | Update::RefreshParameters => {
                self.report.update(cx, msg);
                refresh_partial(cx);
            }
            Update::RefreshReport => {
                self.report.update(cx, Update::RefreshReport);
                refresh_report_only(cx);
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
            Update::OpenNameModal(name) => self.open_modal(cx, InputModal::name(cx, &name)),
            Update::OpenSeedModal(name) => self.open_modal(cx, InputModal::seed(cx, name)),
            Update::OpenPicksModeModal => self.open_modal(cx, InputModal::picks_mode(cx)),
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

        self.toast.render(cx, frame, area);
    }
}

fn refresh_full(cx: &mut Context) {
    cx.report_request_id = cx.report_request_id.wrapping_add(1);
    cx.auto_picks_request_id = cx.auto_picks_request_id.wrapping_add(1);
    cx.manual_picks_request_id = cx.manual_picks_request_id.wrapping_add(1);

    let Some(teams) = cx.teams.clone() else {
        cx.update(Update::ErrorToast("No team data loaded.".to_string()));
        return;
    };

    match Teams::try_from(teams) {
        Ok(teams_soa) => {
            cx.task(Task::RunSimulation {
                request_id: cx.report_request_id,
                teams: Box::new(teams_soa.clone()),
                sigma: cx.sigma,
                iterations: cx.iterations,
                report: cx.report_type,
            });
            cx.task(Task::AutoPicks {
                request_id: cx.auto_picks_request_id,
                teams: Box::new(teams_soa),
                sigma: cx.sigma,
                iterations: cx.iterations,
            });
        }
        Err(e) => {
            cx.update(Update::ErrorToast(format!("Invalid team data: {e}")));
        }
    }
}

fn refresh_partial(cx: &mut Context) {
    cx.report_request_id = cx.report_request_id.wrapping_add(1);
    cx.auto_picks_request_id = cx.auto_picks_request_id.wrapping_add(1);
    cx.manual_picks_request_id = cx.manual_picks_request_id.wrapping_add(1);

    let Some(teams) = cx.teams.clone() else {
        cx.update(Update::ErrorToast("No team data loaded.".to_string()));
        return;
    };

    let teams_soa = match Teams::try_from(teams.clone()) {
        Ok(teams_soa) => teams_soa,
        Err(e) => {
            cx.update(Update::ErrorToast(format!("Invalid team data: {e}")));
            return;
        }
    };

    cx.task(Task::RunSimulation {
        request_id: cx.report_request_id,
        teams: Box::new(teams_soa.clone()),
        sigma: cx.sigma,
        iterations: cx.iterations,
        report: cx.report_type,
    });
    cx.task(Task::AutoPicks {
        request_id: cx.auto_picks_request_id,
        teams: Box::new(teams_soa.clone()),
        sigma: cx.sigma,
        iterations: cx.iterations,
    });

    if cx.picks.iter().all(Option::is_some) {
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
            request_id: cx.manual_picks_request_id,
            teams: Box::new(teams_soa),
            sigma: cx.sigma,
            iterations: cx.iterations,
            three_zero: collect_set(&cx.picks[..2]),
            advanced: collect_set(&cx.picks[2..8]),
            zero_three: collect_set(&cx.picks[8..]),
        });
    }
}

fn refresh_report_only(cx: &mut Context) {
    cx.report_request_id = cx.report_request_id.wrapping_add(1);

    let Some(teams) = cx.teams.clone() else {
        cx.update(Update::ErrorToast("No team data loaded.".to_string()));
        return;
    };

    match Teams::try_from(teams) {
        Ok(teams_soa) => {
            cx.task(Task::RunSimulation {
                request_id: cx.report_request_id,
                teams: Box::new(teams_soa),
                sigma: cx.sigma,
                iterations: cx.iterations,
                report: cx.report_type,
            });
        }
        Err(e) => {
            cx.update(Update::ErrorToast(format!("Invalid team data: {e}")));
        }
    }
}

fn update_data_files(path: PathBuf) -> Update {
    let mut files = Vec::new();

    let iter = match data_updater(&path) {
        Ok(iter) => iter,
        Err(e) => return Update::ErrorToast(format!("Update failed: {e}")),
    };

    for result in iter {
        match result {
            Ok(file) => files.push(file),
            Err(e) => {
                return Update::ErrorToast(format!("Update failed: {e}"));
            }
        }
    }

    Update::DataFilesUpdated { path, files }
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
    cx.update(Update::RefreshFull);
    cx.update(Update::ChangeScreen(Screen::Report));
}

fn set_pick(cx: &mut Context, index: usize, name: Name) {
    cx.manual_picks_request_id = cx.manual_picks_request_id.wrapping_add(1);

    for pick in &mut cx.picks {
        _ = pick.take_if(|pick| pick == &name);
    }

    cx.picks[index] = Some(name);

    if cx.picks.iter().any(Option::is_none) {
        cx.update(Update::ManualPickAssess(
            cx.manual_picks_request_id,
            String::new(),
        ));
        return;
    }

    let Some(teams) = &cx.teams else {
        cx.update(Update::ErrorToast("No team data loaded.".to_string()));
        return;
    };

    let teams_soa = match Teams::try_from(teams.clone()) {
        Ok(teams) => teams,
        Err(e) => {
            cx.update(Update::ErrorToast(format!("Invalid team data: {e}")));
            return;
        }
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
        request_id: cx.manual_picks_request_id,
        teams: Box::new(teams_soa),
        sigma: cx.sigma,
        iterations: cx.iterations,
        three_zero: collect_set(&cx.picks[..2]),
        advanced: collect_set(&cx.picks[2..8]),
        zero_three: collect_set(&cx.picks[8..]),
    });
}
