use ratatui::{
    Frame,
    crossterm::event::{Event, KeyCode, KeyModifiers},
    layout::Rect,
};

use pickems::datatypes::{Map, Teams};

use super::{
    Notify, ReportType, Screen, State, Task, Update, binds,
    entities::{OpenScreen, ReportScreen},
    framework::{Entity, Root},
    tasks,
};

type Context = super::framework::Context<Update, Notify, Task, State>;
type Msg = super::framework::Msg<Update, Notify, Task>;

#[allow(clippy::struct_field_names)]
pub struct App {
    open_screen: OpenScreen,
    report_screen: ReportScreen,
    active_screen: Screen,
}

impl App {
    pub fn new() -> Self {
        let open_screen = OpenScreen::new();
        let report_screen = ReportScreen::new();

        Self {
            open_screen,
            report_screen,
            active_screen: Screen::Open,
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
                Some(Msg::Update(Update::AutoPicks(content)))
            }
        }
    }
}

impl Entity<Update, Notify, Task, State> for App {
    fn dispatch_event(&mut self, cx: &mut Context, event: &Event) -> Option<Msg> {
        match self.active_screen {
            Screen::Open => self.open_screen.dispatch_event(cx, event),
            Screen::Report => self.report_screen.dispatch_event(cx, event),
        }
        .map_or_else(|| self.handle_event(cx, event), Some)
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
        match self.active_screen {
            Screen::Open => self.open_screen.notify(cx, msg),
            Screen::Report => self.report_screen.notify(cx, msg),
        }
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        match msg {
            Update::ChangeScreen(screen) => {
                self.active_screen = screen;
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
            Update::ReportContent(..) | Update::DataOrParams | Update::AutoPicks(..) => {
                self.report_screen.update(cx, msg);
            }
            Update::LoadFileList(..) => self.open_screen.update(cx, msg),
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        match self.active_screen {
            Screen::Open => self.open_screen.render(cx, frame, area),
            Screen::Report => self.report_screen.render(cx, frame, area),
        }
    }
}
