use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::{Constraint, Rect},
    style::Style,
    widgets::{Block, BorderType, Padding, Row, Table, TableState},
};

use crate::app::tui::{Context, Id, Msg, State, Task, Update, binds, framework::Entity};

#[derive(Debug, Clone)]
pub struct ParametersPane {
    state: TableState,
}

impl ParametersPane {
    pub fn new() -> Self {
        Self {
            state: TableState::new().with_selected_cell(Some((0, 1))),
        }
    }

    fn block_style(cx: &Context) -> Style {
        if matches!(cx.report_focus, Id::Parameters) && !cx.modal_open {
            Style::new().blue().bold()
        } else {
            Style::new().bold()
        }
    }

    fn cell_style(cx: &Context) -> Style {
        if matches!(cx.report_focus, Id::Parameters) && !cx.modal_open {
            Style::new().reversed()
        } else {
            Style::new()
        }
    }
}

impl Entity<Update, Task, State> for ParametersPane {
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
            binds::SELECT => match self.state.selected() {
                Some(0) => Some(Msg::Update(Update::OpenReportModal)),
                Some(1) => Some(Msg::Update(Update::OpenIterationsModal)),
                Some(2) => Some(Msg::Update(Update::OpenSigmaModal)),
                _ => None,
            },
            _ => None,
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        let iterations = format!("{}", cx.iterations);
        let sigma = format!("{}", cx.sigma);

        frame.render_stateful_widget(
            Table::new(
                [
                    Row::new(["Report", cx.report_type.as_str()]),
                    Row::new(["Iterations", iterations.as_str()]),
                    Row::new(["Sigma", sigma.as_str()]),
                ],
                [Constraint::Length(10), Constraint::Fill(1)],
            )
            .column_spacing(2)
            .cell_highlight_style(Self::cell_style(cx))
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Self::block_style(cx))
                    .title_style(Self::block_style(cx))
                    .title(format!(
                        "Parameters [{}]",
                        binds::Bind(binds::FOCUS_PARAMETERS)
                    ))
                    .padding(Padding::horizontal(1)),
            ),
            area,
            &mut self.state,
        );
    }
}
