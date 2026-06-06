use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::{Constraint, Rect},
    style::Style,
    text::Line,
    widgets::{Block, BorderType, Cell, Padding, Row, Table, TableState},
};

use pickems::datatypes::Name;

use crate::app::tui::{Context, Id, Msg, State, Task, Update, binds, framework::Entity};

#[derive(Debug, Clone)]
pub struct TeamsPane {
    teams: Vec<(String, Name, String)>,
    state: TableState,
}

impl TeamsPane {
    pub const fn new() -> Self {
        Self {
            teams: Vec::new(),
            state: TableState::new(),
        }
    }

    fn block_style(cx: &Context) -> Style {
        if matches!(cx.report_focus, Id::Teams) && !cx.modal_open {
            Style::new().bold().blue()
        } else {
            Style::new()
        }
    }

    fn cell_style(cx: &Context) -> Style {
        if matches!(cx.report_focus, Id::Teams) && !cx.modal_open {
            Style::new().reversed()
        } else {
            Style::new()
        }
    }
}

impl Entity<Update, Task, State> for TeamsPane {
    fn on_key_press(
        &mut self,
        _cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::UP => {
                self.state.select_previous();
                Some(Msg::Redraw)
            }
            binds::DOWN => {
                self.state.select_next();
                Some(Msg::Redraw)
            }
            binds::LEFT => {
                self.state.select_previous_column();
                Some(Msg::Redraw)
            }
            binds::RIGHT => {
                self.state.select_next_column();
                Some(Msg::Redraw)
            }
            binds::SELECT if let Some((row, col)) = self.state.selected_cell() => {
                let name = self.teams[row].1.clone();

                match col {
                    0 => Some(Msg::Update(Update::OpenSeedModal(name))),
                    1 => Some(Msg::Update(Update::OpenNameModal(name))),
                    2 => Some(Msg::Update(Update::OpenRatingModal(name))),
                    _ => unreachable!(),
                }
            }
            _ => None,
        }
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        if matches!(msg, Update::RefreshFull | Update::RefreshTeamValues)
            && let Some(teams) = &cx.teams
        {
            self.teams = teams
                .iter()
                .map(|(name, team)| {
                    (
                        format!("{} ", team.seed),
                        name.clone(),
                        format!("{} ", team.rating),
                    )
                })
                .collect();

            self.teams
                .sort_unstable_by_key(|(_, name, _)| (teams[name].seed, teams[name].rating));

            if self.state.selected_cell().is_none() {
                self.state.select_cell(Some((0, 0)));
            }
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        frame.render_stateful_widget(
            Table::new(
                self.teams.iter().map(|(seed, name, rating)| {
                    Row::new([
                        Cell::new(Line::from(seed.as_str()).right_aligned()),
                        Cell::new(name.as_str()),
                        Cell::new(Line::from(rating.as_str()).right_aligned()),
                    ])
                }),
                [
                    Constraint::Length(4),
                    Constraint::Fill(1),
                    Constraint::Length(6),
                ],
            )
            .header(Row::new(vec!["Seed", "Team", "Rating"]).style(Style::new().bold()))
            .column_spacing(2)
            .cell_highlight_style(Self::cell_style(cx))
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Self::block_style(cx))
                    .title_style(Self::block_style(cx))
                    .title(format!("Teams [{}]", binds::Bind(binds::FOCUS_TEAMS)))
                    .padding(Padding::horizontal(1)),
            ),
            area,
            &mut self.state,
        );
    }
}
