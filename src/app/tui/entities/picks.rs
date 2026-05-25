use pickems::datatypes::Teams;
use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::Rect,
    style::Style,
    widgets::{Block, BorderType, Paragraph},
};

use crate::app::tui::{Context, Id, Msg, Notify, State, Task, Update, binds, framework::Entity};

#[derive(Debug, Clone)]
pub struct PicksPane {
    content: String,
}

impl PicksPane {
    pub const fn new() -> Self {
        Self {
            content: String::new(),
        }
    }

    fn style(cx: &Context) -> Style {
        if matches!(cx.report_focus, Id::Picks) {
            Style::new().blue().bold()
        } else {
            Style::new().bold()
        }
    }
}

impl Entity<Update, Notify, Task, State> for PicksPane {
    fn on_key_press(
        &mut self,
        _cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::UP => {
                // self.state.select_previous();
                Some(Msg::Redraw)
            }
            binds::DOWN => {
                // self.state.select_next();
                Some(Msg::Redraw)
            }
            _ => None,
        }
    }

    // fn notify(&mut self, cx: &mut Context, msg: Notify) {
    //     if matches!(msg, Notify::Select)
    //         && let Some(i) = self.state.selected()
    //     {
    //         cx.update(Update::LoadDataFile(self.items[i].clone()));
    //     }
    // }

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
            Update::AutoPicks(content) => self.content = content,
            _ => {}
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        frame.render_widget(
            Paragraph::new(self.content.trim()).block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Self::style(cx))
                    .title_style(Self::style(cx))
                    .title(format!("Picks [{}]", binds::Bind(binds::FOCUS_PICKS))),
            ),
            area,
        );
    }
}
