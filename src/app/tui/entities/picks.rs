use std::ops::Range;

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Paragraph},
};

use pickems::datatypes::{Seed, Teams};

use crate::app::tui::{Context, Id, Msg, Notify, State, Task, Update, binds, framework::Entity};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Mode {
    #[default]
    Auto,
    Manual,
}

impl Mode {
    const fn as_str(&self) -> &str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Focus {
    #[default]
    Mode,
    Pick(usize),
}

#[derive(Debug, Clone, Default)]
pub struct PicksPane {
    auto_content: String,
    manual_content: String,
    mode: Mode,
    focus: Focus,
}

impl PicksPane {
    pub fn new() -> Self {
        Self::default()
    }

    fn block_style(cx: &Context) -> Style {
        if matches!(cx.report_focus, Id::Picks) && !cx.modal_open {
            Style::new().blue().bold()
        } else {
            Style::new().bold()
        }
    }

    fn focus_style(&self, cx: &Context, focus: Focus) -> Style {
        if matches!(cx.report_focus, Id::Picks) && !cx.modal_open && self.focus == focus {
            Style::new().reversed()
        } else {
            Style::new()
        }
    }
}

impl Entity<Update, Notify, Task, State> for PicksPane {
    fn on_key_press(
        &mut self,
        cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        let Some(teams) = &cx.teams else { return None };

        match self.focus {
            Focus::Mode => match (modifiers, code) {
                binds::DOWN if self.mode == Mode::Manual => self.focus = Focus::Pick(0),
                binds::LEFT => self.mode = Mode::Auto,
                binds::RIGHT => self.mode = Mode::Manual,
                _ => return None,
            },
            Focus::Pick(n) => match (modifiers, code) {
                binds::UP if n > 0 => self.focus = Focus::Pick(n - 1),
                binds::UP => self.focus = Focus::Mode,
                binds::DOWN if n < 9 => self.focus = Focus::Pick(n + 1),
                binds::LEFT => {
                    let mut seed = cx.picks[n].as_ref().map_or_else(
                        || Seed::new(16),
                        |name| teams[name].seed.wrapping_decrement(),
                    );

                    let new_name = loop {
                        let new_name = teams.iter().find_map(|(name, team)| {
                            if team.seed == seed { Some(name) } else { None }
                        })?;

                        if !cx
                            .picks
                            .iter()
                            .any(|name| name.as_ref().is_some_and(|n| n == new_name))
                        {
                            break new_name;
                        }

                        seed = seed.wrapping_decrement();
                    };

                    return Some(Msg::Update(Update::SetPick {
                        index: n,
                        name: new_name.clone(),
                    }));
                }
                binds::RIGHT => {
                    let mut seed = cx.picks[n].as_ref().map_or_else(
                        || Seed::new(1),
                        |name| teams[name].seed.wrapping_increment(),
                    );

                    let new_name = loop {
                        let new_name = teams.iter().find_map(|(name, team)| {
                            if team.seed == seed { Some(name) } else { None }
                        })?;

                        if !cx
                            .picks
                            .iter()
                            .any(|name| name.as_ref().is_some_and(|n| n == new_name))
                        {
                            break new_name;
                        }

                        seed = seed.wrapping_increment();
                    };

                    return Some(Msg::Update(Update::SetPick {
                        index: n,
                        name: new_name.clone(),
                    }));
                }
                _ => return None,
            },
        }

        Some(Msg::Redraw)
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        match msg {
            Update::DataOrParams if let Some(teams) = &cx.teams => {
                if let Ok(teams_soa) = Teams::try_from(teams.clone()) {
                    cx.task(Task::AutoPicks {
                        teams: Box::new(teams_soa),
                        sigma: cx.sigma,
                        iterations: cx.iterations,
                    });
                } else {
                    cx.notify(Notify::Todo);
                }
            }
            Update::AutoPickAssess(content) => self.auto_content = content,
            Update::ManualPickAssess(content) => self.manual_content = content,
            _ => {}
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Self::block_style(cx))
            .title_style(Self::block_style(cx))
            .title(format!("Picks [{}]", binds::Bind(binds::FOCUS_PICKS)));

        let [top, _, bottom] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Fill(1),
        ])
        .areas(block.inner(area));

        frame.render_widget(block, area);
        frame.render_widget(
            Line::from_iter([
                Span::from("Mode: "),
                Span::from(self.mode.as_str()).style(self.focus_style(cx, Focus::Mode)),
            ]),
            top,
        );

        match self.mode {
            Mode::Auto => {
                frame.render_widget(Paragraph::new(self.auto_content.trim()), bottom);
            }
            Mode::Manual => {
                let team_lines = |range: Range<usize>| {
                    range.map(|i| {
                        Line::from(cx.picks[i].as_ref().map_or("-", |name| name.as_str()))
                            .style(self.focus_style(cx, Focus::Pick(i)))
                    })
                };

                let mut picks_text = Text::default();
                picks_text.push_line("3-0 picks:");

                for line in team_lines(0..2) {
                    picks_text.push_line(line);
                }

                picks_text.push_line("");
                picks_text.push_line("3-1 or 3-2 picks:");

                for line in team_lines(2..8) {
                    picks_text.push_line(line);
                }

                picks_text.push_line("");
                picks_text.push_line("0-3 picks:");

                for line in team_lines(8..10) {
                    picks_text.push_line(line);
                }

                let [picks_area, content_area] =
                    Layout::vertical([Constraint::Length(16), Constraint::Fill(1)]).areas(bottom);

                frame.render_widget(picks_text, picks_area);
                frame.render_widget(Paragraph::new(self.manual_content.trim()), content_area);
            }
        }
    }
}
