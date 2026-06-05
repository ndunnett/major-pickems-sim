use ratatui::{
    Frame,
    crossterm::event::{Event, KeyCode, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::Span,
};

use crate::app::tui::{
    Context, Id, Msg, Screen, State, Task, Update, binds,
    entities::{ParametersPane, PicksPane, ReportPane, TeamsPane},
    framework::Entity,
};

pub struct ReportScreen {
    teams: TeamsPane,
    picks: PicksPane,
    parameters: ParametersPane,
    report: ReportPane,
}

impl ReportScreen {
    pub fn new() -> Self {
        let teams = TeamsPane::new();
        let picks = PicksPane::new();
        let parameters = ParametersPane::new();
        let report = ReportPane::new();

        Self {
            teams,
            picks,
            parameters,
            report,
        }
    }
}

impl Entity<Update, Task, State> for ReportScreen {
    fn dispatch_event(&mut self, cx: &mut Context, event: &Event) -> Option<Msg> {
        match cx.report_focus {
            Id::Teams => self.teams.dispatch_event(cx, event),
            Id::Picks => self.picks.dispatch_event(cx, event),
            Id::Report => self.report.dispatch_event(cx, event),
            Id::Parameters => self.parameters.dispatch_event(cx, event),
        }
        .map_or_else(|| self.handle_event(cx, event), Some)
    }

    fn on_key_press(
        &mut self,
        cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::FOCUS_TEAMS => {
                cx.report_focus = Id::Teams;
                Some(Msg::Redraw)
            }
            binds::FOCUS_PICKS => {
                cx.report_focus = Id::Picks;
                Some(Msg::Redraw)
            }
            binds::FOCUS_PARAMETERS => {
                cx.report_focus = Id::Parameters;
                Some(Msg::Redraw)
            }
            binds::FOCUS_REPORT => {
                cx.report_focus = Id::Report;
                Some(Msg::Redraw)
            }
            binds::FOCUS_NEXT => {
                cx.report_focus = cx.report_focus.next();
                Some(Msg::Redraw)
            }
            binds::FOCUS_PREV => {
                cx.report_focus = cx.report_focus.prev();
                Some(Msg::Redraw)
            }
            binds::OPEN_SCREEN => {
                cx.update(Update::ChangeScreen(Screen::Open));
                Some(Msg::Redraw)
            }
            binds::SAVE if cx.teams.is_some() => Some(Msg::Update(Update::OpenSaveModal)),
            _ => None,
        }
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        match msg {
            Update::DataOrParams => {
                self.teams.update(cx, msg.clone());
                self.picks.update(cx, msg);
            }
            Update::ReportContent(..) => self.report.update(cx, msg),
            Update::AutoPickAssess(..) => self.picks.update(cx, msg),
            _ => match cx.report_focus {
                Id::Teams => self.teams.update(cx, msg),
                Id::Picks => self.picks.update(cx, msg),
                Id::Report => self.report.update(cx, msg),
                Id::Parameters => self.parameters.update(cx, msg),
            },
        }
    }

    #[inline]
    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        let [window, keybar] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(area);

        let [top, bottom] =
            Layout::vertical([Constraint::Length(24), Constraint::Fill(1)]).areas(window);

        let [left, _, right] = Layout::horizontal([
            Constraint::Fill(1),
            Constraint::Length(1),
            Constraint::Fill(1),
        ])
        .areas(top);

        let [left_upper, left_lower] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(5)]).areas(left);

        self.teams.render(cx, frame, left_upper);
        self.parameters.render(cx, frame, left_lower);
        self.picks.render(cx, frame, right);
        self.report.render(cx, frame, bottom);

        let open = Span::from(format!(
            "{} [{}]",
            cx.opened
                .as_ref()
                .and_then(|path| path.file_name())
                .map_or_else(|| "Open".into(), |path| path.to_string_lossy()),
            binds::Bind(binds::OPEN_SCREEN)
        ))
        .style(Style::new().blue().bold());

        let quit = Span::from(format!("Quit [{}]", binds::Bind(binds::QUIT)))
            .style(Style::new().blue().bold());

        let save = Span::from(format!("Save [{}]", binds::Bind(binds::SAVE)))
            .style(Style::new().blue().bold());

        let [open_area, save_area, _, quit_area] = Layout::horizontal([
            Constraint::Length(open.width() as u16),
            Constraint::Length(save.width() as u16),
            Constraint::Fill(1),
            Constraint::Length(quit.width() as u16),
        ])
        .spacing(4)
        .areas(keybar);

        frame.render_widget(open, open_area);
        frame.render_widget(save, save_area);
        frame.render_widget(quit, quit_area);
    }
}
