use ratatui::{
    Frame,
    crossterm::event::{Event, KeyCode, KeyModifiers},
    layout::Rect,
};

use pickems::datatypes::{Index, Map, Name, Set, Teams};

use crate::app::tui::{
    Notify, ReportType, Screen, State, Task, Update, binds,
    entities::{InputModal, OpenScreen, ReportScreen},
    framework::{Entity, Root},
    tasks,
};

type Context = crate::app::tui::framework::Context<Update, Notify, Task, State>;
type Msg = crate::app::tui::framework::Msg<Update, Notify, Task>;

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
}

impl Root<Update, Notify, Task, State> for App {
    const MAX_FPS: u64 = 120;

    fn handle_task(task: Task) -> Option<Msg> {
        match task {
            Task::UpdateData { path } => {
                if tasks::update_data(&path).is_ok_and(|it| it.count() > 0) {
                    Some(Msg::Update(Update::LoadFileList(path)))
                } else {
                    Some(Msg::Notify(Notify::Todo))
                }
            }
            Task::RunSimulation {
                teams,
                sigma,
                iterations,
                report,
            } => {
                let content = tasks::run_simulation(*teams, sigma, iterations, report);
                Some(Msg::Update(Update::ReportContent(content)))
            }
            Task::AutoPicks {
                teams,
                sigma,
                iterations,
            } => {
                let content = tasks::run_simulation(*teams, sigma, iterations, ReportType::Picks);
                Some(Msg::Update(Update::AutoPickAssess(content)))
            }
            Task::ManualPicks {
                teams,
                sigma,
                iterations,
                three_zero,
                advanced,
                zero_three,
            } => {
                let content = tasks::assess_picks(
                    *teams, sigma, iterations, three_zero, advanced, zero_three,
                );
                Some(Msg::Update(Update::ManualPickAssess(content)))
            }
        }
    }
}

impl Entity<Update, Notify, Task, State> for App {
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

    fn notify(&mut self, cx: &mut Context, msg: Notify) {
        match self.active {
            Screen::Open => self.open.notify(cx, msg),
            Screen::Report => self.report.notify(cx, msg),
        }
    }

    #[allow(clippy::too_many_lines)]
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
            Update::LoadDataFile(path) => {
                if let Ok(teams) = Map::parse_toml(path.clone())
                    && let Ok(teams_soa) = Teams::try_from(teams.clone())
                {
                    cx.opened = Some(path);
                    cx.teams = Some(teams);
                    cx.update(Update::DataOrParams);
                    cx.update(Update::ChangeScreen(Screen::Report));

                    cx.task(Task::RunSimulation {
                        teams: Box::new(teams_soa),
                        sigma: cx.sigma,
                        iterations: cx.iterations,
                        report: cx.report_type,
                    });
                } else {
                    cx.notify(Notify::Todo);
                }
            }
            Update::SetPick { index, name } => {
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
            Update::ReportContent(..)
            | Update::AutoPickAssess(..)
            | Update::ManualPickAssess(..)
            | Update::PicksMode(..) => {
                self.report.update(cx, msg);
            }
            Update::DataOrParams => {
                self.report.update(cx, msg);

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
                    cx.notify(Notify::Todo);
                }
            }
            Update::LoadFileList(..) => self.open.update(cx, msg),
            Update::CloseModal => {
                self.input_modal = None;
                cx.modal_open = false;
            }
            Update::OpenReportModal => {
                self.input_modal = Some(InputModal::report(cx));
                cx.modal_open = true;
            }
            Update::OpenIterationsModal => {
                self.input_modal = Some(InputModal::iterations(cx));
                cx.modal_open = true;
            }
            Update::OpenSigmaModal => {
                self.input_modal = Some(InputModal::sigma(cx));
                cx.modal_open = true;
            }
            Update::OpenRatingModal(name) => {
                self.input_modal = Some(InputModal::rating(cx, name));
                cx.modal_open = true;
            }
            Update::OpenNameModal(name) => {
                self.input_modal = Some(InputModal::name(name));
                cx.modal_open = true;
            }
            Update::OpenSeedModal(name) => {
                self.input_modal = Some(InputModal::seed(cx, name));
                cx.modal_open = true;
            }
            Update::OpenPicksModeModal => {
                self.input_modal = Some(InputModal::picks_mode());
                cx.modal_open = true;
            }
            Update::OpenPickSelectModal(n) => {
                self.input_modal = Some(InputModal::pick_select(cx, n));
                cx.modal_open = true;
            }
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
