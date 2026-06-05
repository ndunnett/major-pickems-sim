use std::path::PathBuf;

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::Rect,
    style::Style,
    widgets::{Block, BorderType, List, ListState},
};

use crate::app::tui::{Context, Msg, State, Task, Update, binds, framework::Entity, tasks};

#[derive(Debug, Clone)]
pub struct FilePicker {
    state: ListState,
    items: Vec<PathBuf>,
}

impl FilePicker {
    pub fn new() -> Self {
        Self {
            state: ListState::default(),
            items: Vec::with_capacity(8),
        }
    }
}

impl Entity<Update, Task, State> for FilePicker {
    fn on_key_press(
        &mut self,
        _cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::SELECT if let Some(i) = self.state.selected() => {
                Some(Msg::Update(Update::LoadDataFile(self.items[i].clone())))
            }
            binds::UP => {
                self.state.select_previous();
                Some(Msg::Redraw)
            }
            binds::DOWN => {
                self.state.select_next();
                Some(Msg::Redraw)
            }
            _ => None,
        }
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        if let Update::LoadFileList(path) = msg {
            if let Ok(iter) = tasks::load_file_list(&path) {
                self.items.clear();
                self.items.extend(iter);
                self.items.sort_by(|a, b| b.cmp(a));

                if self.state.selected().is_none() {
                    self.state.select_first();
                }
            } else {
                cx.update(Update::Todo);
            }
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        let items = self
            .items
            .iter()
            .filter_map(|path| path.file_name())
            .map(|path| path.to_string_lossy());

        let path = cx.path.canonicalize().unwrap_or_else(|_| cx.path.clone());

        frame.render_stateful_widget(
            List::new(items)
                .block(
                    Block::bordered()
                        .border_type(BorderType::Rounded)
                        .title_style(Style::new().bold())
                        .title(path.to_string_lossy()),
                )
                .highlight_style(Style::new().reversed())
                .highlight_symbol(">")
                .repeat_highlight_symbol(true),
            area,
            &mut self.state,
        );
    }
}
